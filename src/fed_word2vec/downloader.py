"""
Fed transcript downloading utilities.

Handles downloading of FOMC statements/minutes, press conferences, and Chair speeches
(Powell, Yellen, Bernanke). Saves raw HTML for downstream text extraction.
"""

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tqdm import tqdm

BASE_MONPOL = "https://www.federalreserve.gov/monetarypolicy/"
BASE_SPEECH = "https://www.federalreserve.gov/newsevents/speech/"
TARGET_SPEAKERS = ["Powell", "Yellen", "Bernanke"]


def create_session(user_agent: str = "FedResearchDownloader/1.0 (contact: research@example.com)") -> requests.Session:
    """Create a configured requests session."""
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name: str) -> str:
    """Remove special characters from filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def normalize_date(text: str | None) -> str | None:
    """Parse and normalize date to ISO format."""
    if not text:
        return None
    try:
        dt = dateparser.parse(text, fuzzy=True, default=datetime(1900, 1, 1))
        return dt.date().isoformat()
    except Exception:
        return None


def fetch_calendar_pages(session: requests.Session) -> list[str]:
    """Collect all FOMC calendar archive pages (1994–present)."""
    start = urljoin(BASE_MONPOL, "fomccalendars.htm")
    resp = session.get(start)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    pages = {start}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"fomc.*\.htm", href, re.I):
            pages.add(urljoin(BASE_MONPOL, href))
    return sorted(pages)


def parse_fomc_page(session: requests.Session, page_url: str) -> list[dict]:
    """Extract Statement/Minutes/Press Conf links from a single calendar page."""
    resp = session.get(page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        lower = text.lower()
        if not any(k in lower for k in ["statement", "minutes", "press conference"]):
            continue
        doc_type = (
            "press_conference"
            if "press conference" in lower
            else "minutes"
            if "minutes" in lower
            else "statement"
        )
        href = urljoin(page_url, a["href"])
        tr = a.find_parent("tr")
        meeting = None
        if tr:
            first_td = tr.find("td")
            if first_td:
                meeting = first_td.get_text(" ", strip=True)
        date_iso = normalize_date(meeting or text)
        rows.append(
            {
                "meeting": meeting,
                "date_iso": date_iso,
                "doc_type": doc_type,
                "title": text,
                "url": href,
                "source_page": page_url,
            }
        )
    return rows


def deduplicate_records(records: list[dict]) -> list[dict]:
    """Drop duplicate links by (url, doc_type) to handle messy archives."""
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for rec in records:
        key = (rec["url"], rec["doc_type"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique


def download_documents(
    session: requests.Session,
    records: list[dict],
    root: Path,
    delay: float = 0.4
) -> None:
    """Download FOMC documents to disk."""
    ensure_dir(root)
    for rec in tqdm(records, desc="Downloading FOMC docs"):
        doc_dir = ensure_dir(root / rec["doc_type"])
        parsed = urlparse(rec["url"])
        tail = sanitize_filename(Path(parsed.path).name or "index.html")
        date_hint = sanitize_filename(rec.get("meeting") or rec.get("date_iso") or "unknown")
        fname = f"{date_hint}-{tail}"
        path = doc_dir / fname
        if path.exists():
            continue
        try:
            r = session.get(rec["url"], timeout=30)
            r.raise_for_status()
            path.write_bytes(r.content)
            time.sleep(delay)
        except Exception as exc:
            print(f"Failed {rec['url']} -> {exc}")


def collect_fomc_documents(session: requests.Session, output_dir: Path) -> list[dict]:
    """
    Scrape and download all FOMC documents.
    
    Args:
        session: Configured requests session
        output_dir: Directory to save downloaded HTML files
        
    Returns:
        List of record dicts with metadata about downloaded documents
    """
    pages = fetch_calendar_pages(session)
    records: list[dict] = []
    for page in tqdm(pages, desc="Parsing calendar pages"):
        try:
            records.extend(parse_fomc_page(session, page))
        except Exception as exc:
            print(f"Skip {page}: {exc}")
    unique_records = deduplicate_records(records)
    download_documents(session, unique_records, output_dir)
    return unique_records


def collect_speeches_from_yearly(
    session: requests.Session,
    start_year: int,
    end_year: int
) -> list[dict]:
    """Try yearly archives. Extract speaker from URL filename."""
    rows: list[dict] = []
    for year in tqdm(range(start_year, end_year + 1), desc="Speeches by year (yearly archives)", leave=False):
        url = f"{BASE_SPEECH}{year}-speeches.htm"
        resp = session.get(url)
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        anchors = soup.find_all("a", href=True)
        for a in anchors:
            href = a["href"]
            if "/newsevents/speech/" not in href or not href.lower().endswith(".htm"):
                continue
            full_url = urljoin(url, href)
            filename = Path(href).stem.lower()
            speaker = next((s for s in TARGET_SPEAKERS if s.lower() in filename), None)
            if not speaker:
                continue
            title = a.get_text(" ", strip=True)
            parent = a.find_parent(["li", "article", "div", "p"])
            context = title
            if parent:
                context += " " + parent.get_text(" ", strip=True)
            date_match = re.search(r"[A-Z][a-z]+ \d{1,2}, \d{4}", context)
            pub_date = date_match.group(0) if date_match else str(year)
            date_iso = normalize_date(pub_date)
            rows.append(
                {
                    "date": pub_date,
                    "date_iso": date_iso,
                    "speaker": speaker,
                    "title": title,
                    "url": full_url,
                }
            )
    return rows


def collect_speeches_from_index(
    session: requests.Session,
    start_year: int = 1994,
    end_year: int | None = None
) -> list[dict]:
    """Scrape main speech index and filter by speaker and year."""
    if end_year is None:
        end_year = datetime.now(timezone.utc).year
    rows: list[dict] = []
    
    candidate_urls = [
        urljoin(BASE_SPEECH, "index.htm"),
        urljoin(BASE_SPEECH, "speech.htm"),
        urljoin(BASE_SPEECH, "speeches.htm"),
        "https://www.federalreserve.gov/newsevents/",
    ]
    
    for index_url in candidate_urls:
        try:
            print(f"Trying {index_url}...")
            resp = session.get(index_url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            anchors = soup.find_all("a", href=True)
            found_count = 0
            for a in anchors:
                href = a["href"]
                if "/newsevents/speech/" not in href or not href.lower().endswith(".htm"):
                    continue
                full_url = urljoin(index_url, href)
                filename = Path(href).stem.lower()
                speaker = next((s for s in TARGET_SPEAKERS if s.lower() in filename), None)
                if not speaker:
                    continue
                title = a.get_text(" ", strip=True)
                parent = a.find_parent(["li", "article", "div", "tr", "p"])
                context = title
                if parent:
                    context += " " + parent.get_text(" ", strip=True)
                date_match = re.search(r"[A-Z][a-z]+ \d{1,2}, (\d{4})", context)
                if date_match:
                    pub_date = date_match.group(0)
                    year = int(date_match.group(1))
                    if year < start_year or year > end_year:
                        continue
                    date_iso = normalize_date(pub_date)
                else:
                    pub_date = "unknown"
                    date_iso = None
                rows.append(
                    {
                        "date": pub_date,
                        "date_iso": date_iso,
                        "speaker": speaker,
                        "title": title,
                        "url": full_url,
                    }
                )
                found_count += 1
            if found_count > 0:
                print(f"Found {found_count} speeches at {index_url}")
                break
        except Exception as exc:
            print(f"  {index_url} failed: {exc}")
    
    return rows


def download_speeches(
    session: requests.Session,
    records: list[dict],
    root: Path,
    delay: float = 0.4
) -> None:
    """Download speech HTML files organized by speaker."""
    ensure_dir(root)
    for rec in tqdm(records, desc="Downloading speeches"):
        speaker_dir = ensure_dir(root / rec["speaker"])
        date_hint = sanitize_filename(rec.get("date_iso") or rec.get("date") or "unknown")
        tail = sanitize_filename(Path(urlparse(rec["url"]).path).name or "speech.html")
        fname = f"{date_hint}-{tail}"
        path = speaker_dir / fname
        if path.exists():
            continue
        try:
            r = session.get(rec["url"], timeout=30)
            r.raise_for_status()
            path.write_bytes(r.content)
            time.sleep(delay)
        except Exception as exc:
            print(f"Failed {rec['url']} -> {exc}")


def collect_speeches(
    session: requests.Session,
    output_dir: Path,
    start_year: int = 1994,
    end_year: int | None = None
) -> list[dict]:
    """
    Download speeches for Powell, Yellen, Bernanke using multiple fallbacks.
    
    Args:
        session: Configured requests session
        output_dir: Directory to save downloaded HTML files
        start_year: First year to collect
        end_year: Last year to collect (defaults to current year)
        
    Returns:
        List of record dicts with metadata about downloaded speeches
    """
    if end_year is None:
        end_year = datetime.now(timezone.utc).year
    print("Attempting yearly archives...")
    all_rows = collect_speeches_from_yearly(session, start_year, end_year)
    if not all_rows:
        print("Yearly archives empty; falling back to main index...")
        all_rows = collect_speeches_from_index(session, start_year, end_year)
    download_speeches(session, all_rows, output_dir)
    return all_rows
