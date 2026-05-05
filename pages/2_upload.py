"""Page 2 — File Upload

Wires up the data pipeline: takes uploaded files, runs them through
extractor.py and chunker.py, and stores the resulting chunks in
st.session_state["chunks"] for downstream steps (embedder, clusterer).

Contract (see docs/data_contracts.md):
    Output:
        st.session_state["chunks"]: list[dict]
            Each dict has keys: chunk_id, text, filename, source_type

Constraints (see CLAUDE.md):
    - Soft cap of ~500 chunks per session (Streamlit Cloud memory)
    - Source types: "interview" | "review" | "ticket" | "usability"
"""
import streamlit as st

from pipeline.extractor import extract_text
from pipeline.chunker import chunk_text


# Must match ALLOWED_SOURCE_TYPES in pipeline/extractor.py and chunker.py
SOURCE_TYPES = ["interview", "review", "ticket", "usability"]

# Soft cap from CLAUDE.md
MAX_CHUNKS_SOFT = 500


st.set_page_config(page_title="Upload — Discovery Lens", page_icon="📂")
st.title("📂 Upload your discovery files")

# --- Goal guard --------------------------------------------------------------
# Per the multi-page flow: user must set a goal (page 1) before uploading.
if not st.session_state.get("goal"):
    st.warning("Please set your product goal first.")
    st.page_link("pages/1_goal.py", label="← Go to Goal page", icon="🎯")
    st.stop()

# --- Goal context strip ------------------------------------------------------
product_name = st.session_state.get("product_name", "—")
goal = st.session_state.get("goal", "")
st.markdown(f"**Product:** {product_name}")
st.markdown(f"**Goal:** {goal}")
st.divider()


# --- Upload form -------------------------------------------------------------
source_type = st.selectbox(
    "Source type for these files",
    options=SOURCE_TYPES,
    help="Tag every file in this batch with the same source type. "
         "Upload different source types in separate batches.",
)

uploaded_files = st.file_uploader(
    "Upload discovery files (PDF, TXT, CSV, DOCX)",
    type=["pdf", "txt", "csv", "docx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more files to begin.")
    st.stop()

st.success(f"{len(uploaded_files)} file(s) ready — tagged as **{source_type}**.")


# --- Run pipeline ------------------------------------------------------------
if st.button("▶ Run pipeline", type="primary"):

    all_chunks: list[dict] = []
    failed_files: list[tuple[str, str]] = []  # (filename, error_message)

    progress = st.progress(0.0, text="Starting…")
    total = len(uploaded_files)

    for idx, file in enumerate(uploaded_files, start=1):
        progress.progress(
            (idx - 1) / total,
            text=f"Processing {file.name} ({idx}/{total})",
        )

        try:
            raw_text = extract_text(file, source_type)
            if not raw_text.strip():
                failed_files.append((file.name, "Extracted text is empty"))
                continue

            chunks = chunk_text(
                raw_text=raw_text,
                filename=file.name,
                source_type=source_type,
            )
            all_chunks.extend(chunks)

        except Exception as exc:
            # Don't let one bad file kill the whole batch.
            failed_files.append((file.name, str(exc)))

    progress.progress(1.0, text="Done")

    # --- Persist results to session_state ------------------------------------
    # Append rather than overwrite so users can upload multiple source-type
    # batches in one session (e.g. interviews first, then reviews).
    existing_chunks = st.session_state.get("chunks", [])
    st.session_state["chunks"] = existing_chunks + all_chunks

    # --- Summary -------------------------------------------------------------
    total_chunks = len(st.session_state["chunks"])

    if all_chunks:
        st.success(
            f"Added {len(all_chunks)} chunks from this batch. "
            f"Total chunks in session: **{total_chunks}**."
        )
    else:
        st.error("No chunks were produced from this batch.")

    if failed_files:
        with st.expander(f"⚠️ {len(failed_files)} file(s) failed", expanded=True):
            for name, err in failed_files:
                st.write(f"- **{name}** — {err}")

    if total_chunks > MAX_CHUNKS_SOFT:
        st.warning(
            f"You have {total_chunks} chunks. The recommended soft limit is "
            f"{MAX_CHUNKS_SOFT} per session — beyond that, performance and "
            f"clustering quality may degrade."
        )

    # --- Next-step nav -------------------------------------------------------
    if all_chunks:
        st.page_link(
            "pages/3_results.py",
            label="Continue to results →",
            icon="🌳",
        )
