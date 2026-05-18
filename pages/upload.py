import streamlit as st
import pypdf
from pipeline.extractor import extract_text as extract
from pipeline.chunker import chunk_text as chunk
from pipeline.embedder import embed_chunks as embed, embed_text
from pipeline.clusterer import cluster
from pipeline.odi_scorer import score_clusters as score
from pipeline.llm import build_ost
from pipeline.source_map import build_source_map

st.set_page_config(page_title="Discovery Lens", layout="centered")

SOURCE_TYPES = ["interview", "review", "ticket", "usability", "social", "internal"]
SOURCE_DESCRIPTIONS = {
    "interview":  "User interview transcripts",
    "review":     "App store or G2/Capterra reviews",
    "ticket":     "Customer support tickets",
    "usability":  "Usability test session notes",
    "social":     "Reddit threads, Twitter/social media posts",
    "internal":   "Sales call notes, CS summaries, internal docs",
}

goal = st.session_state.get("goal", "")
context_block = st.session_state.get("context_block", "")
if goal:
    st.markdown(
        f'<div style="background:#EEEDFE;border-left:4px solid #534AB7;border-radius:6px;'
        f'padding:10px 16px;font-size:13px;color:#534AB7;margin-bottom:16px;">'
        f'<span style="font-weight:600;">Goal:</span> {goal}</div>',
        unsafe_allow_html=True,
    )

if "pipeline_stage" not in st.session_state:
    st.session_state["pipeline_stage"] = None


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Upload
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state["pipeline_stage"] is None:

    st.title("Upload your discovery data")
    st.caption("Supported formats: PDF, TXT, CSV, DOCX · Max 500 chunks total")
    st.divider()

    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "txt", "csv", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.subheader("Tag each file")
        file_configs = []
        for f in uploaded_files:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{f.name}**")
            with col2:
                src = st.selectbox(
                    "Type", SOURCE_TYPES,
                    key=f"source_{f.name}",
                    label_visibility="collapsed",
                )
            st.caption(SOURCE_DESCRIPTIONS[src])
            file_configs.append({"file": f, "source_type": src})

        # ── Optional stakeholder context ──────────────────────────────────────
        _MAX_CONTEXT_WORDS = 500
        with st.expander("Stakeholder context (optional)"):
            st.caption(
                "Add constraints, stakeholder priorities, or known assumptions "
                "that should influence the analysis. Max 500 words."
            )
            context_typed = st.text_area(
                "Type context",
                placeholder=(
                    "e.g. Engineering has flagged real-time notifications as out "
                    "of scope this quarter. The PM has already committed to "
                    "improving the onboarding flow…"
                ),
                height=140,
                label_visibility="collapsed",
                key="_context_typed_raw",
            )
            context_pdf = st.file_uploader(
                "Or upload a context PDF",
                type=["pdf"],
                key="_context_pdf",
            )
            context_pdf_text = ""
            if context_pdf is not None:
                reader = pypdf.PdfReader(context_pdf)
                context_pdf_text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                ).strip()

            combined = "\n\n".join(
                part for part in [context_typed.strip(), context_pdf_text] if part
            )
            words = combined.split()
            if len(words) > _MAX_CONTEXT_WORDS:
                st.warning(
                    f"Context truncated from {len(words)} to {_MAX_CONTEXT_WORDS} words."
                )
                st.session_state["context_block"] = " ".join(words[:_MAX_CONTEXT_WORDS])
            else:
                st.session_state["context_block"] = combined
            if combined:
                st.caption(f"{min(len(words), _MAX_CONTEXT_WORDS)} / {_MAX_CONTEXT_WORDS} words")

        st.divider()
        if st.button("Analyse clusters →", type="primary"):
            try:
                with st.status("Analysing your files…", expanded=True) as status:

                    st.write("Extracting text…")
                    all_chunks = []
                    for fc in file_configs:
                        raw = extract(fc["file"], fc["source_type"])
                        all_chunks.extend(chunk(raw, fc["file"].name, fc["source_type"]))

                    if not all_chunks:
                        status.update(label="Nothing to analyse.", state="error")
                        st.error("⚠️ No text could be extracted. Make sure your files contain readable text and try again.")
                        st.stop()

                    st.session_state["chunks"] = all_chunks
                    st.write(f"✓ {len(all_chunks)} chunks extracted")

                    st.write("Embedding…")
                    embeddings = embed(all_chunks)
                    st.session_state["embeddings"] = embeddings
                    st.write("✓ Embeddings ready")

                    st.write("Clustering…")
                    clusters = cluster(all_chunks, embeddings)
                    st.session_state["clusters"] = clusters

                    if not clusters:
                        status.update(label="Clustering found nothing.", state="error")
                        st.error("⚠️ No clusters could be formed. Try uploading more content — at least 3–5 documents work best.")
                        st.stop()

                    st.write("Scoring…")
                    goal_embedding = embed_text(goal) if goal else None
                    scored = score(clusters, all_chunks, goal_embedding, embeddings)
                    st.session_state["scored_clusters"] = scored
                    st.write(f"✓ {len(clusters)} clusters · {len(all_chunks)} chunks scored")

                    status.update(label="Clustering complete — review below.", state="complete")

                st.session_state["pipeline_stage"] = "clustered"
                st.rerun()

            except Exception as e:
                st.error(f"⚠️ Something went wrong: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Cluster review
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state["pipeline_stage"] == "clustered":

    clusters   = st.session_state.get("clusters", [])
    all_chunks = st.session_state.get("chunks", [])
    scored     = st.session_state.get("scored_clusters", [])
    score_map  = {s["cluster_id"]: s for s in scored}

    source_types_found = sorted({c.get("source_type", "") for c in all_chunks if c.get("source_type")})
    n_types = len(source_types_found)

    st.title("Review clusters")
    st.caption(
        f"{len(all_chunks)} chunks · {len(clusters)} clusters · "
        f"{n_types} source type{'s' if n_types != 1 else ''}"
    )

    # ── Source diversity warnings ──────────────────────────────────────────────
    if n_types == 1:
        st.warning(
            f"📂 **Single source type ({source_types_found[0].capitalize()} only).** "
            f"Evidence robustness scores will be lower. Consider adding interviews, tickets, or reviews."
        )
    elif n_types < 3:
        st.info(
            f"ℹ️ **{n_types} source types found ({', '.join(source_types_found)}).** "
            f"Adding a third source type would improve evidence robustness scores."
        )

    if len(clusters) < 3:
        st.warning(
            f"🔢 **Only {len(clusters)} cluster{'s' if len(clusters) > 1 else ''} found.** "
            f"Uploading more files usually surfaces more distinct themes."
        )

    st.divider()

    # ── Cluster summaries ──────────────────────────────────────────────────────
    for cl in clusters:
        cid  = cl["cluster_id"]
        sc   = score_map.get(cid, {})
        ps   = sc.get("priority_score") or 0
        reps = cl.get("representative_chunks", [])[:2]
        types = sorted({c.get("source_type", "") for c in all_chunks
                        if c.get("chunk_id") in cl.get("all_chunk_ids", [])})

        with st.expander(
            f"Cluster {cid + 1}  ·  priority {ps:.2f}  ·  "
            f"{len(cl.get('all_chunk_ids', []))} chunks  ·  {', '.join(types) or 'unknown'}",
            expanded=(cid < 3),
        ):
            if not reps:
                st.caption("No representative chunks available.")
            for rep in reps:
                st.markdown(
                    f'<div style="border-left:3px solid #AFA9EC;padding:6px 12px;'
                    f'margin-bottom:8px;font-size:13px;font-style:italic;">'
                    f'"{rep["text"][:220]}{"…" if len(rep["text"]) > 220 else ""}"'
                    f'<div style="font-size:10px;color:#888;margin-top:4px;">'
                    f'{rep.get("source_type","").capitalize()} · {rep.get("filename","")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Action buttons ─────────────────────────────────────────────────────────
    col_back, col_go = st.columns([1, 2])

    with col_back:
        if st.button("← Upload different files"):
            for key in ("pipeline_stage", "clusters", "chunks", "embeddings",
                        "scored_clusters", "ost", "source_map"):
                st.session_state.pop(key, None)
            st.rerun()

    with col_go:
        if st.button("Generate OST →", type="primary"):
            try:
                with st.status("Building Opportunity-Solution Tree…", expanded=True) as status:

                    st.write("Calling LLM…")
                    try:
                        ost = build_ost(clusters, scored, goal, context_block)
                    except RuntimeError as llm_err:
                        err_str = str(llm_err).lower()
                        if "rate limit" in err_str or "429" in err_str:
                            status.update(label="Groq rate limit hit.", state="error")
                            st.error("⚠️ Groq API rate limit reached. Wait a minute and try again, or reduce the number of files.")
                        else:
                            status.update(label="LLM step failed.", state="error")
                            st.error(f"⚠️ The LLM step failed: {llm_err}")
                        st.stop()

                    st.session_state["ost"] = ost
                    st.write(f"✓ {len(ost.get('opportunities', []))} opportunities found")

                    st.session_state["source_map"] = build_source_map(all_chunks, clusters)
                    status.update(label="OST ready!", state="complete")

                st.session_state["pipeline_stage"] = "done"
                st.rerun()

            except Exception as e:
                st.error(f"⚠️ Something went wrong: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Done
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state["pipeline_stage"] == "done":
    n_opps   = len(st.session_state.get("ost", {}).get("opportunities", []))
    n_chunks = len(st.session_state.get("chunks", []))
    st.success(f"✓ {n_opps} opportunities found across {n_chunks} chunks.")

    col_back, col_go = st.columns([1, 2])
    with col_back:
        if st.button("← Upload different files"):
            for key in ("pipeline_stage", "clusters", "chunks", "embeddings",
                        "scored_clusters", "ost", "source_map"):
                st.session_state.pop(key, None)
            st.rerun()
    with col_go:
        if st.button("Next →", type="primary"):
            st.session_state["goal"] = goal
            st.session_state["pipeline_stage"] = None
            st.switch_page("pages/results.py")
