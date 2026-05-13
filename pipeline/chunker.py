"""
chunker.py — raw text → chunks

Part of the Discovery Lens pipeline. Takes the raw text produced by
extractor.py and splits it into chunks of 2–4 sentences, each tagged with
source metadata for downstream clustering and traceability.

Contract (see docs/data_contracts.md):
    Input:
        raw_text: str
        filename: str
        source_type: str       # "interview" | "review" | "ticket" | "usability"
    Output:
        chunks: list[dict]     # one dict per chunk with keys:
            chunk_id, text, filename, source_type
"""

from __future__ import annotations

import re
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize

# Allowed source types — must match extractor.py and CLAUDE.md.
# Expanded May 13 2026: added "social" and "internal" per docs/decisions.md.
ALLOWED_SOURCE_TYPES = {
    "interview",
    "review",
    "ticket",
    "usability",
    "social",
    "internal",
}

# How many sentences go into a single chunk. Contract says 2–4.
SENTENCES_PER_CHUNK = 3

# Minimum sentence length (in characters) — shorter "sentences" are usually
# artefacts of sentence splitting (e.g. "OK.", "Yeah.") and add noise to clusters.
MIN_SENTENCE_LENGTH = 10


# Module-level flag so we only attempt to download punkt_tab once per process,
# even if chunk_text() is called many times in a single Streamlit session.
_PUNKT_READY = False


def _ensure_nltk_punkt() -> None:
    """
    Download the nltk 'punkt_tab' tokenizer data if it's not already available.

    Called lazily on first chunk_text() invocation so importing the module
    doesn't hit the network. Idempotent — repeated calls are cheap.

    Raises
    ------
    RuntimeError
        If punkt_tab is not installed locally and the download fails (e.g. no
        internet). Clearer than the default nltk LookupError deep in the stack.
    """
    global _PUNKT_READY
    if _PUNKT_READY:
        return

    try:
        nltk.data.find("tokenizers/punkt_tab")
        _PUNKT_READY = True
        return
    except LookupError:
        pass

    # Not installed — try to download it.
    try:
        nltk.download("punkt_tab", quiet=True)
        nltk.data.find("tokenizers/punkt_tab")  # confirm the download worked
        _PUNKT_READY = True
    except Exception as e:
        raise RuntimeError(
            "nltk 'punkt_tab' is not installed and could not be downloaded. "
            "Run `python -c \"import nltk; nltk.download('punkt_tab')\"` once "
            f"with internet access. Original error: {e}"
        ) from e


def _safe_filename(filename: str) -> str:
    """
    Convert a filename into a chunk_id-safe slug.

    Strips the extension, lowercases, and replaces any non-alphanumeric
    character with an underscore. Collapses repeated underscores.

    Examples
    --------
    >>> _safe_filename("Interview 01.txt")
    'interview_01'
    >>> _safe_filename("reviews_revolut.csv")
    'reviews_revolut'
    """
    stem = Path(filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem)
    return slug.strip("_")


def chunk_text(raw_text: str, filename: str, source_type: str) -> list[dict]:
    """
    Split raw text into chunks of 2–4 sentences with source metadata.

    Parameters
    ----------
    raw_text : str
        Full text extracted from one file (output of extractor.extract_text).
    filename : str
        Original filename (e.g. "interview_01.txt"). Used to build chunk_ids.
    source_type : str
        One of ALLOWED_SOURCE_TYPES.

    Returns
    -------
    list[dict]
        One dict per chunk with keys:
          - chunk_id: "{safe_filename}_{zero_padded_index}" (e.g. "interview_01_001")
          - text: the chunk's text
          - filename: passed through
          - source_type: passed through

        Returns an empty list if raw_text is empty or contains no usable sentences.

    Raises
    ------
    ValueError
        If source_type is not in ALLOWED_SOURCE_TYPES.
    """
    # Validate source_type — fail fast, same rule as extractor.py
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError(
            f"Invalid source_type '{source_type}'. "
            f"Must be one of: {sorted(ALLOWED_SOURCE_TYPES)}"
        )

    # Empty input → empty output. Don't crash.
    if not raw_text or not raw_text.strip():
        return []

    # Make sure nltk data is available (downloads on first run only).
    _ensure_nltk_punkt()

    # Split into sentences with nltk. Handles abbreviations, multiple languages,
    # and weird punctuation better than a plain regex.
    sentences = sent_tokenize(raw_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    # Prefer sentences above MIN_SENTENCE_LENGTH to avoid noise like "OK." or "Yeah.",
    # but fall back to the unfiltered list if filtering would wipe everything out
    # (e.g. a short app-store review made entirely of 3–5 word sentences).
    filtered = [s for s in sentences if len(s) >= MIN_SENTENCE_LENGTH]
    sentences = filtered if filtered else sentences

    # Group sentences into chunks of SENTENCES_PER_CHUNK.
    # The last chunk may be shorter if the sentence count is not divisible.
    safe = _safe_filename(filename)
    chunks: list[dict] = []

    for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
        group = sentences[i : i + SENTENCES_PER_CHUNK]
        chunk_text_value = " ".join(group)

        # 1-based, zero-padded to 3 digits → "001", "002", ...
        # If a file ever exceeds 999 chunks (unlikely but possible for huge
        # CSVs), the index will keep growing — "1000", "1001" — which still
        # sorts correctly lexicographically for our purposes.
        index = (i // SENTENCES_PER_CHUNK) + 1
        chunk_id = f"{safe}_{index:03d}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text_value,
                "filename": filename,
                "source_type": source_type,
            }
        )

    return chunks
