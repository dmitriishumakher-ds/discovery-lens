"""
extractor.py — file → raw text

Part of the Discovery Lens pipeline. Takes a Streamlit UploadedFile object
and returns a plain string of its text content.

Contract (see docs/data_contracts.md):
    Input:
        file: UploadedFile     # Streamlit UploadedFile
        source_type: str       # "interview" | "review" | "ticket" | "usability"
    Output:
        raw_text: str          # full extracted text
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pandas as pd
from pypdf import PdfReader
from docx import Document

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile


# Allowed source types — keep in sync with odi_scorer.py and CLAUDE.md.
# Expanded May 13 2026: added "social" and "internal" per docs/decisions.md.
ALLOWED_SOURCE_TYPES = {
    "interview",
    "review",
    "ticket",
    "usability",
    "social",
    "internal",
}


def extract_text(file: "UploadedFile", source_type: str) -> str:
    """
    Extract plain text from an uploaded file.

    Parameters
    ----------
    file : UploadedFile
        Streamlit UploadedFile object (has .name, .read(), .getvalue()).
    source_type : str
        One of ALLOWED_SOURCE_TYPES. Used by downstream chunker.

    Returns
    -------
    str
        Full extracted text. Never None; returns "" if the file is empty.

    Raises
    ------
    ValueError
        If source_type is not in ALLOWED_SOURCE_TYPES, or the file extension
        is not supported.
    """
    # Validate source_type early — fail fast before doing any work
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError(
            f"Invalid source_type '{source_type}'. "
            f"Must be one of: {sorted(ALLOWED_SOURCE_TYPES)}"
        )

    filename = file.name.lower()

    if filename.endswith(".pdf"):
        return _extract_pdf(file)
    if filename.endswith(".docx"):
        return _extract_docx(file)
    if filename.endswith(".csv"):
        return _extract_csv(file)
    if filename.endswith(".txt"):
        return _extract_txt(file)

    raise ValueError(
        f"Unsupported file extension: '{file.name}'. "
        f"Supported: .pdf, .docx, .csv, .txt"
    )


# ---------------------------------------------------------------------------
# Private helpers — one per file type
# ---------------------------------------------------------------------------


def _extract_pdf(file: "UploadedFile") -> str:
    """Extract text from a PDF using pypdf. Joins pages with double newlines."""
    reader = PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def _extract_docx(file: "UploadedFile") -> str:
    """Extract text from a DOCX using python-docx. Joins paragraphs with newlines."""
    # python-docx needs a file-like object, not an UploadedFile directly
    doc = Document(io.BytesIO(file.getvalue()))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def _extract_csv(file: "UploadedFile") -> str:
    """
    Extract text from a CSV.

    Strategy: concatenate all text-like columns, row by row, with " | " separator
    between columns and "\n" between rows. This preserves row boundaries for
    downstream chunking.
    """
    df = pd.read_csv(file)

    # Keep only object (string) columns — numeric columns like ratings add noise
    text_cols = df.select_dtypes(include="object").columns.tolist()

    if not text_cols:
        return ""

    # Build one line per row by joining text cells with " | "
    lines = []
    for _, row in df[text_cols].iterrows():
        cells = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
        if cells:
            lines.append(" | ".join(cells))

    return "\n".join(lines)


def _extract_txt(file: "UploadedFile") -> str:
    """Extract text from a plain TXT file. UTF-8 with a latin-1 fallback."""
    raw_bytes = file.getvalue()
    try:
        return raw_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        # Fallback for older files or mixed encodings
        return raw_bytes.decode("latin-1", errors="replace").strip()
