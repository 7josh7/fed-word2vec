"""
Fed Word2Vec - Data pipeline utilities.

This package contains core logic for downloading and processing Federal Reserve
communications for Word2Vec training.

Modules:
    downloader: Functions for scraping FOMC documents and Fed Chair speeches
    text_cleaner: HTML parsing and text cleaning/filtering utilities
"""

from .downloader import (
    create_session,
    collect_fomc_documents,
    collect_speeches,
    sanitize_filename,
    normalize_date,
)

from .text_cleaner import (
    extract_text,
    tokenize,
    process_file,
    process_tree,
    recommend_guard_from_profile,
    compute_removed_lines,
)

__all__ = [
    # downloader
    "create_session",
    "collect_fomc_documents",
    "collect_speeches",
    "sanitize_filename",
    "normalize_date",
    # text_cleaner
    "extract_text",
    "tokenize",
    "process_file",
    "process_tree",
    "recommend_guard_from_profile",
    "compute_removed_lines",
]
