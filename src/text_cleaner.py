"""
Text extraction and cleaning utilities for Fed documents.

Parses raw HTML into cleaned text with line-level filtering,
optimized for FOMC minutes and Fed Chair speeches.
"""

import re
import statistics
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, Comment

# Tags to strip from HTML
STRIP_TAGS = {"script", "style", "nav", "footer", "header", "form", "aside", "table"}

# CSS selectors to try for main content extraction (in priority order)
CONTENT_SELECTORS = [
    "div#article",
    "div#content",
    "div.col-xs-12.col-sm-8",
    "main",
    "article",
    "div#contentwrapper",
]

# Boilerplate phrases to remove
BOILERPLATE = [
    "Board of Governors of the Federal Reserve System",
    "Back to top",
    "Return to text",
    "Skip to main content",
    "Accessibility",
    "Contact Us",
    "Last Update:",
    "Home |",
    "| Privacy",
]

# Patterns for lines to remove (obvious garbage)
REMOVE_LINE_PATTERNS = [
    r"^\s*PRESENT\s*:.*$",
    r"^\s*ATTENDEES?\s*:.*$",
    r"^\s*ATTENDING\s*:.*$",
    r"^\s*Attended.*session.*$",
    r"Return to text",
    r"\d+\.\s*Return to text",
    r"Footnote\s*\d+",
    r"^\s*\d+\s*$",
    r"^\s*[a-z]\.\s*$",
    r"^\s*\(\d+\)\s*$",
    r"^\s*\([a-z]\)\s*$",
    r"^[\s\-_=]+$",
    r"^.*https?://.*$",
    r"^.*\bPDF\b.*$",
    r"^.*\.pdf\b.*$",
]

# Narrative keywords to preserve (domain-aware)
NARRATIVE_KEYWORDS = {
    # macro + policy
    "inflation", "policy", "rate", "funds", "market", "economic", "growth", "financial",
    "committee", "participants", "staff", "balance", "sheet", "reserve", "credit", "risk", "employment",
    "output", "gdp", "prices", "labor", "wage", "supply", "demand", "investment", "spending",
    "unemployment", "ioer", "iorb", "rrp", "soma", "balance sheet", "runoff", "transmission",
    "forecast", "projection", "estimate", "assessment", "decision", "rationale", "tools",
    # facilities & operations
    "mbs", "mortgage-backed", "mortgage backed", "treasury", "treasuries",
    "qe", "qt", "quantitative easing", "quantitative tightening",
    "repo", "reverse repo", "repurchase",
    "discount window", "primary credit", "secondary credit",
    "standing repo facility", "srf",
    "facility", "facilities",
    "swap", "swap lines", "fima",
    "liquidity",
    "open market", "desk", "soma operations",
}

# Verb pattern to detect narrative sentences
VERB_PATTERN = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|will|would|should|can|could|may|might|does|do|did|\w+ing|\w+ed)\b",
    re.IGNORECASE
)

# Default removal guard threshold
DEFAULT_REMOVAL_GUARD = 0.4


def _pick_main_node(soup: BeautifulSoup):
    """Try to isolate the primary content block; fall back to <body>."""
    for selector in CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node:
            return node
    return soup.body or soup


def _remove_boilerplate(text: str) -> str:
    """Remove common boilerplate phrases."""
    for phrase in BOILERPLATE:
        text = text.replace(phrase, " ")
    return text


def _remove_line_artifacts(text: str) -> str:
    """Remove footnote/navigation artifacts while preserving lines."""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        skip = False
        for pattern in REMOVE_LINE_PATTERNS:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                skip = True
                break
        if not skip:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _line_is_narrative(line: str) -> bool:
    """Check if line contains narrative content worth keeping."""
    low = line.lower()
    if any(k in low for k in NARRATIVE_KEYWORDS):
        return True
    if VERB_PATTERN.search(line):
        words = re.findall(r"[A-Za-z']+", line)
        if len(words) >= 6:
            return True
    return False


def _line_is_garbage(line: str) -> bool:
    """Check if line is likely garbage (roster, list, etc.)."""
    for pattern in REMOVE_LINE_PATTERNS:
        if re.match(pattern, line.strip(), re.IGNORECASE):
            return True
    words = [w for w in re.findall(r"[A-Za-z']+", line)]
    if len(words) >= 6:
        comma_count = line.count(",")
        if comma_count >= 3 and not VERB_PATTERN.search(line):
            return True
    return False


def extract_text(html: str, is_fomc: bool = False, removal_guard: float | None = None) -> str:
    """
    Extract and clean text from HTML.
    
    Args:
        html: Raw HTML content
        is_fomc: If True, apply stricter FOMC-specific cleaning
        removal_guard: Max allowed removal ratio before fallback (None disables guard)
        
    Returns:
        Cleaned text content
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    main = _pick_main_node(soup)

    # Remove unwanted HTML tags
    for tag in main.find_all(list(STRIP_TAGS)):
        tag.decompose()

    # Preserve newlines during initial extraction
    text = main.get_text("\n", strip=True)

    # Basic boilerplate removal
    text = _remove_boilerplate(text)

    if is_fomc:
        # Minimal cleanup first to form baseline
        baseline = _remove_line_artifacts(text)
        base_chars = len(baseline)
        base_lines = baseline.split("\n")

        # Line-level filtering: keep narrative or not-garbage lines
        kept_lines = []
        for ln in base_lines:
            if _line_is_narrative(ln):
                kept_lines.append(ln)
            elif not _line_is_garbage(ln):
                kept_lines.append(ln)
        cleaned = "\n".join(kept_lines)

        # Sanity guard: if >X% removed, fallback to light cleaning only
        guard = removal_guard if removal_guard is not None else DEFAULT_REMOVAL_GUARD
        if guard is not None and base_chars:
            removed_ratio = 1.0 - (len(cleaned) / base_chars)
            if removed_ratio > guard:
                print(f"[Guard] Fallback to light cleaning (removed {removed_ratio*100:.1f}%, guard {guard*100:.0f}%)")
                cleaned = baseline
        text = cleaned
    else:
        # Speeches: light cleaning only
        text = _remove_line_artifacts(text)

    # Final normalization
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Simple tokenization to lowercase alphanumeric tokens."""
    return [t.lower() for t in re.findall(r"[A-Za-z0-9']+", text)]


def compute_removed_lines(baseline: str, cleaned: str) -> list[str]:
    """Return lines removed by filtering, preserving order and duplicates."""
    b_lines = baseline.split("\n")
    c_counts = Counter(cleaned.split("\n"))
    removed = []
    for ln in b_lines:
        if c_counts.get(ln, 0) > 0:
            c_counts[ln] -= 1
        else:
            removed.append(ln)
    return removed


def process_file(
    path: Path,
    input_root: Path,
    output_root: Path,
    is_fomc: bool = False,
    removal_guard: float | None = None
) -> tuple[int, int, Counter] | None:
    """
    Process a single HTML file: extract text, tokenize, and save.
    
    Args:
        path: Path to HTML file
        input_root: Root directory of input files (for relative path calculation)
        output_root: Root directory for output text files
        is_fomc: If True, apply FOMC-specific cleaning
        removal_guard: Max allowed removal ratio
        
    Returns:
        Tuple of (char_count, token_count, token_counter) or None if failed
    """
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        html = path.read_text(encoding="latin-1", errors="ignore")

    cleaned = extract_text(html, is_fomc=is_fomc, removal_guard=removal_guard)

    if not cleaned:
        return None

    tokens = tokenize(cleaned)

    rel = path.relative_to(input_root)
    out_path = (output_root / rel).with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned, encoding="utf-8")

    return len(cleaned), len(tokens), Counter(tokens)


def process_tree(
    input_root: Path,
    output_root: Path,
    is_fomc: bool = False,
    removal_guard: float | None = None,
    removed_collector: list[str] | None = None
) -> tuple[int, int, int, Counter, list[float]]:
    """
    Process all HTML files in a directory tree.
    
    Args:
        input_root: Root directory containing HTML files
        output_root: Root directory for output text files
        is_fomc: If True, apply FOMC-specific cleaning
        removal_guard: Max allowed removal ratio
        removed_collector: Optional list to collect removed lines for analysis
        
    Returns:
        Tuple of (file_count, total_chars, total_tokens, token_counter, removal_ratios)
    """
    paths = [
        p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in {".htm", ".html"}
    ]
    total_files = 0
    total_chars = 0
    total_tokens = 0
    counter: Counter = Counter()
    removal_ratios: list[float] = []

    for p in paths:
        try:
            html = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            html = p.read_text(encoding="latin-1", errors="ignore")
        
        baseline = extract_text(html, is_fomc=False)
        cleaned_for_stats = extract_text(html, is_fomc=is_fomc, removal_guard=removal_guard)
        
        if baseline:
            ratio = 1.0 - (len(cleaned_for_stats) / len(baseline))
            removal_ratios.append(max(0.0, min(1.0, ratio)))
            
            if removed_collector is not None and is_fomc:
                removed_lines = compute_removed_lines(baseline, cleaned_for_stats)
                if removed_lines:
                    removed_collector.append(f"=== File: {p}")
                    removed_collector.extend(removed_lines)
                    removed_collector.append("")

        result = process_file(p, input_root, output_root, is_fomc=is_fomc, removal_guard=removal_guard)
        if not result:
            continue
        chars, tokens, local_counter = result
        total_files += 1
        total_chars += chars
        total_tokens += tokens
        counter.update(local_counter)

    return total_files, total_chars, total_tokens, counter, removal_ratios


def recommend_guard_from_profile(ratios: list[float], default: float = DEFAULT_REMOVAL_GUARD) -> float:
    """
    Recommend a removal guard threshold based on corpus statistics.
    
    Args:
        ratios: List of removal ratios from corpus profiling
        default: Default guard value if no ratios provided
        
    Returns:
        Recommended guard threshold (clamped 30-60%)
    """
    if not ratios:
        return default
    med = statistics.median(ratios)
    mean = statistics.fmean(ratios)
    p90 = sorted(ratios)[int(0.9 * (len(ratios) - 1))]
    guard = med + 0.10
    guard = max(0.30, min(0.60, guard))
    print("Corpus removal profile (FOMC):")
    print(f"  Mean: {mean*100:.1f}% | Median: {med*100:.1f}% | 90th: {p90*100:.1f}%")
    print(f"  Recommended guard: {guard*100:.0f}% (clamped 30–60%)")
    return guard


def print_removed_text(baseline: str, cleaned: str, max_lines: int = 40) -> None:
    """Print removed lines for debugging."""
    removed = compute_removed_lines(baseline, cleaned)
    print(f"--- REMOVED (first {max_lines} lines) ---")
    for ln in removed[:max_lines]:
        print(ln)
    print(f"(showing {min(max_lines, len(removed))}/{len(removed)} removed lines)")
