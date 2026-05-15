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

# T-05: minimum chunk quality thresholds.
# A chunk with only 1 short sentence embeds poorly (the embedding has no
# semantic context to anchor on) and tends to be absorbed by the nearest
# large cluster, inflating its importance. We require both:
#   - At least MIN_SENTENCES_PER_CHUNK substantive sentences
#   - At least MIN_TOKENS_PER_CHUNK whitespace tokens (~words)
# Whitespace tokeniser is intentional — keeps the module dependency-free.
MIN_SENTENCES_PER_CHUNK = 2
MIN_TOKENS_PER_CHUNK = 15

# T-07 (benchmark in notebooks/chunk_size_benchmark.ipynb): token-window chunking
# outperformed sentence-window on KMeans+cosine silhouette by +31.6% (0.1103 vs 0.0838)
# on the Revolut synthetic corpus. Token-based chunks are also more uniform in size,
# which helps embedding stability and reduces T-05 filter drop from 11.1% to 2.4%.
#
# Sentence-based logic is kept as a tested fallback — flip USE_TOKEN_CHUNKING to False
# if a future dataset distribution rules differently.
USE_TOKEN_CHUNKING = True
TOKENS_PER_CHUNK = 80
TOKEN_OVERLAP = 20

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


def _is_chunk_substantive(text: str) -> bool:
    """
    T-05: filter out impoverished chunks before they reach clustering.

    Returns True if the chunk meets both quality thresholds:
      - At least MIN_SENTENCES_PER_CHUNK sentences (counted by punctuation)
      - At least MIN_TOKENS_PER_CHUNK whitespace tokens

    Short, vague chunks (single sentences, half-sentences, lone exclamations)
    embed poorly. They get absorbed into the nearest large cluster and inflate
    its importance score — which then biases priority_score downstream.
    Filtering them at this stage is cheaper and more honest than trying to
    recover from a noisy cluster later.

    Counting sentences: split on .?! and count non-empty fragments. This is
    approximate but cheap, and consistent with how we feed text to nltk.
    Counting tokens: simple whitespace split. Not a true token count
    (tokenisers vary), but a stable floor that requires no dependencies.
    """
    sentence_fragments = [s.strip() for s in re.split(r"[.?!]+", text) if s.strip()]
    if len(sentence_fragments) < MIN_SENTENCES_PER_CHUNK:
        return False

    token_count = len(text.split())
    if token_count < MIN_TOKENS_PER_CHUNK:
        return False

    return True


def _chunk_by_tokens(
    raw_text: str,
    target_tokens: int = TOKENS_PER_CHUNK,
    overlap: int = TOKEN_OVERLAP,
) -> list[str]:
    """
    T-07: token-window chunker (sliding window with overlap).

    Splits text on whitespace and groups tokens into chunks of approximately
    `target_tokens` words, advancing by `target_tokens - overlap` each step.
    Overlap preserves semantic continuity across chunk boundaries — useful
    when a sentence spans the cut point.

    Why whitespace tokens instead of nltk tokens:
      - No new dependency (sentence_transformers tokeniser would be best but
        loads a model just to count tokens, which is overkill here).
      - Whitespace count is close enough to true token count for chunking
        purposes — embeddings are robust to ±10% size variance.

    Why this beats sentence-based chunking:
      - Uniform chunk size → more stable embeddings → tighter clusters.
      - Higher T-05 filter survival (2.4% vs 11.1% in benchmark).

    Returns a list of chunk strings. Very short trailing chunks (<10 tokens)
    are dropped to avoid noise at end-of-document boundaries.
    """
    tokens = raw_text.split()
    if len(tokens) < 10:
        return []

    chunks: list[str] = []
    step = max(target_tokens - overlap, 1)
    for i in range(0, len(tokens), step):
        group = tokens[i : i + target_tokens]
        if len(group) < 10:
            # Trailing group too short to carry semantic signal.
            continue
        chunks.append(" ".join(group))
    return chunks


def _chunk_by_sentences(raw_text: str) -> list[str]:
    """
    Legacy sentence-window chunker (T-07 fallback path).

    Used when USE_TOKEN_CHUNKING is False. Kept because:
      - Some future dataset may favour sentence boundaries (e.g. structured
        legal text, dialogue transcripts where sentence breaks carry meaning).
      - Easy A/B comparison if the team wants to re-benchmark.

    Logic identical to the pre-T-07 production behaviour: nltk sentence split,
    drop very short sentences, group into windows of SENTENCES_PER_CHUNK.
    """
    _ensure_nltk_punkt()
    sentences = sent_tokenize(raw_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []
    filtered = [s for s in sentences if len(s) >= MIN_SENTENCE_LENGTH]
    sentences = filtered if filtered else sentences

    chunks: list[str] = []
    for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
        group = sentences[i : i + SENTENCES_PER_CHUNK]
        chunks.append(" ".join(group))
    return chunks


def chunk_text(raw_text: str, filename: str, source_type: str) -> list[dict]:
    """
    Split raw text into chunks with source metadata.

    Strategy is controlled by USE_TOKEN_CHUNKING (default True after T-07):
      - True  → 80-token sliding window with 20-token overlap (T-07 winner)
      - False → 3-sentence non-overlapping window (legacy, pre-T-07)

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

        Returns an empty list if raw_text is empty or contains no usable text.

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

    # T-07: choose chunking strategy
    if USE_TOKEN_CHUNKING:
        chunk_strings = _chunk_by_tokens(raw_text)
    else:
        chunk_strings = _chunk_by_sentences(raw_text)

    if not chunk_strings:
        return []

    # Wrap strings in chunk dicts with source metadata
    safe = _safe_filename(filename)
    chunks: list[dict] = []
    for index, chunk_text_value in enumerate(chunk_strings, start=1):
        chunk_id = f"{safe}_{index:03d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text_value,
                "filename": filename,
                "source_type": source_type,
            }
        )

    # --- T-05: filter impoverished chunks before dedup ---
    # Single-sentence or sub-15-token chunks embed poorly. Drop them here
    # so they don't get absorbed by a large cluster downstream.
    chunks_before_filter = len(chunks)
    chunks = [c for c in chunks if _is_chunk_substantive(c["text"])]
    filtered_out = chunks_before_filter - len(chunks)
    if filtered_out > 0:
        # Logged at chunker level; the caller (2_upload.py) may surface to UI.
        # No formal logger to avoid a new dependency — print is intentional
        # and only fires when filtering actually removed something.
        print(
            f"[chunker:{filename}] T-05 filtered {filtered_out} impoverished chunks "
            f"({chunks_before_filter} → {len(chunks)})"
        )

    # --- T-01: deduplicate on chunk text within this file ---
    # Catches near-identical documents (e.g. someone uploads the same
    # content under two different filenames). The MD5 guard in 2_upload.py
    # catches exact-byte duplicates; this catches text duplicates.
    seen_texts: set[str] = set()
    unique_chunks: list[dict] = []
    for chunk in chunks:
        if chunk["text"] not in seen_texts:
            seen_texts.add(chunk["text"])
            unique_chunks.append(chunk)

    return unique_chunks
