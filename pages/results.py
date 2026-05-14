import json
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Discovery Lens", layout="wide")

RISK_COLOR = {"low": "#639922", "medium": "#EF9F27", "high": "#E24B4A"}

DEMO_OST = {
    "goal": "Help product teams at B2B SaaS companies turn raw discovery into prioritised action within one working day",
    "opportunities": [
        {
            "jtbd": "When I finish user interviews, I want to spot recurring themes quickly, so I can brief my team without spending a day in spreadsheets.",
            "cluster_id": 0, "importance": 0.42, "satisfaction": 0.27, "source_type_diversity": 0.75,
            "odi_score": 0.31, "evidence_robustness": 0.64, "priority_score": 0.44,
            "solutions": [
                {"label": "Auto-cluster on upload", "assumptions": [{"text": "Users upload consistently", "risk": "low"}]},
                {"label": "JTBD summary card", "assumptions": [{"text": "LLM output is reliable", "risk": "medium"}]},
                {"label": "AI synthesis report", "assumptions": [{"text": "Users trust AI output", "risk": "high"}]},
            ]
        },
        {
            "jtbd": "When I present findings to stakeholders, I want to show which quotes back each insight, so I can defend prioritisation decisions.",
            "cluster_id": 1, "importance": 0.35, "satisfaction": 0.40, "source_type_diversity": 0.50,
            "odi_score": 0.21, "evidence_robustness": 0.45, "priority_score": 0.31,
            "solutions": [
                {"label": "Quote traceability panel", "assumptions": [{"text": "Source map is complete", "risk": "low"}]},
                {"label": "Exportable evidence map", "assumptions": [{"text": "Export format is agreed", "risk": "medium"}]},
            ]
        },
        {
            "jtbd": "When I'm juggling multiple data sources, I want a single place to upload everything, so I can avoid stitching files manually.",
            "cluster_id": 2, "importance": 0.31, "satisfaction": 0.50, "source_type_diversity": 0.50,
            "odi_score": 0.16, "evidence_robustness": 0.43, "priority_score": 0.27,
            "solutions": [
                {"label": "Multi-file drag-drop", "assumptions": [{"text": "File formats are supported", "risk": "low"}]},
                {"label": "Source type tagging", "assumptions": [{"text": "Users tag correctly", "risk": "low"}]},
            ]
        },
        {
            "jtbd": "When I share an OST with my team, I want them to understand the scoring logic, so I can get buy-in without a long explanation.",
            "cluster_id": 3, "importance": 0.22, "satisfaction": 0.54, "source_type_diversity": 0.25,
            "odi_score": 0.10, "evidence_robustness": 0.24, "priority_score": 0.16,
            "solutions": [
                {"label": "Inline score tooltip", "assumptions": [{"text": "Tooltip is discoverable", "risk": "medium"}]},
                {"label": "JSON export", "assumptions": [{"text": "Team reads JSON", "risk": "low"}]},
            ]
        },
    ]
}

DEMO_SOURCE_MAP = {
    "interview_asana_03_007": {"text": "I usually end up copy-pasting into a giant Notion doc and then spending two days trying to find the common threads by hand.", "filename": "interview_asana_03.txt", "source_type": "interview", "cluster_id": 0},
    "g2_asana_019_002": {"text": "The hardest part of discovery isn't collecting data — it's making sense of it fast enough to be useful in sprint planning.", "filename": "g2_asana_019.csv", "source_type": "review", "cluster_id": 0},
    "interview_asana_01_012": {"text": "We have themes, but proving they're real to engineering always requires going back to every original note. It's exhausting.", "filename": "interview_asana_01.txt", "source_type": "interview", "cluster_id": 0},
    "ticket_asana_005_003": {"text": "Stakeholders keep asking me to justify my roadmap decisions — I need a way to show the evidence trail quickly.", "filename": "tickets_asana.csv", "source_type": "ticket", "cluster_id": 1},
    "interview_asana_02_008": {"text": "I spent three hours building a slide just to show why one feature matters more than another. There has to be a better way.", "filename": "interview_asana_02.txt", "source_type": "interview", "cluster_id": 1},
}

# ── DATA ──────────────────────────────────────────────────────────────────────
ost = st.session_state.get("ost") or DEMO_OST
source_map = st.session_state.get("source_map") or DEMO_SOURCE_MAP
product_name = st.session_state.get("product_name", "")
is_demo = not st.session_state.get("ost")

goal_display = st.session_state.get("goal") or ost.get("goal", "")

opportunities = sorted(
    ost.get("opportunities", []),
    key=lambda o: o.get("priority_score") or 0,
    reverse=True,
)

if "selected_opp" not in st.session_state:
    st.session_state["selected_opp"] = 0

# ── HEADER ────────────────────────────────────────────────────────────────────
col_title, col_export = st.columns([4, 1])
with col_title:
    if product_name:
        st.caption(f"Product · {product_name}")
    st.markdown("## Discovery results")
    if is_demo:
        st.caption("Showing demo data — run the pipeline on the upload screen to see your results.")
with col_export:
    st.download_button(
        label="Export JSON ↓",
        data=json.dumps(ost, indent=2),
        file_name="discovery_lens_ost.json",
        mime="application/json",
    )

if goal_display:
    st.markdown(
        f'<div style="background:#EEEDFE;border-left:4px solid #534AB7;border-radius:6px;'
        f'padding:10px 16px;font-size:13px;color:#534AB7;margin-bottom:4px;">'
        f'<span style="font-weight:600;">Goal:</span> {goal_display}'
        f'{"  —  <em>" + product_name + "</em>" if product_name else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── QUALITY NOTICES ───────────────────────────────────────────────────────────
if not is_demo:
    notices = []

    # 1 — source type diversity
    source_types_found = {c.get("source_type") for c in source_map.values() if c.get("source_type")}
    if len(source_types_found) == 1:
        only_type = next(iter(source_types_found)).capitalize()
        notices.append(
            f"📂 **Single source type detected ({only_type} only).** "
            f"Evidence robustness scores are lower when all files share the same source type. "
            f"Upload interviews, support tickets, or app reviews alongside your {only_type.lower()} "
            f"files to get a more reliable ranking."
        )

    # 2 — low cluster count
    cluster_count = len(st.session_state.get("clusters", []))
    if 0 < cluster_count < 3:
        notices.append(
            f"🔢 **Only {cluster_count} cluster{'s' if cluster_count > 1 else ''} found.** "
            f"The pipeline needs more content to surface distinct opportunity themes. "
            f"Try uploading at least 5–10 documents across different source types."
        )

    # 3 — LLM fallback was triggered
    if ost.get("_meta", {}).get("used_fallback"):
        notices.append(
            "🔁 **LLM fallback used.** The primary model returned invalid JSON, so the pipeline "
            "retried with the smaller fallback model. JTBD statements may be less detailed than usual."
        )

    for msg in notices:
        st.warning(msg)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_ost, tab_matrix, tab_heatmap = st.tabs(["Opportunities", "Priority matrix", "Evidence heatmap"])

# ── TAB 1: OPPORTUNITIES ──────────────────────────────────────────────────────
with tab_ost:

    # Section label
    st.markdown(
        f'<div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-bottom:12px;">'
        f'Ranked Opportunities · {len(opportunities)} cluster{"s" if len(opportunities) != 1 else ""} found'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Opportunity cards (3 per row) ─────────────────────────────────────────
    SOURCE_DOT = {"interview": "#534AB7", "review": "#0C447C", "ticket": "#639922",
                  "usability": "#EF9F27", "sales": "#E24B4A", "other": "#888888"}

    cols = st.columns(min(len(opportunities), 3))
    for idx, opp in enumerate(opportunities):
        col = cols[idx % 3]
        ps  = opp.get("priority_score") or 0
        odi = opp.get("odi_score") or 0
        rob = opp.get("evidence_robustness") or 0
        is_top = idx == 0
        top_border = "border-top:4px solid #534AB7;" if is_top else "border-top:4px solid #E0DFDB;"
        rank_bg    = "#534AB7" if is_top else "#F1EFE8"
        rank_color = "#FFFFFF" if is_top else "#888888"
        rank_label = f"#{idx+1} · Priority" if is_top else f"#{idx+1}"

        def score_bar(label, value, fill_color):
            pct = int(value * 100)
            return (
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">'
                f'<div style="font-size:10px;color:#888;width:115px;flex-shrink:0;">{label}</div>'
                f'<div style="flex:1;height:6px;background:#F1EFE8;border-radius:4px;overflow:hidden;">'
                f'<div style="height:100%;width:{pct}%;background:{fill_color};border-radius:4px;"></div>'
                f'</div>'
                f'<div style="font-size:10px;font-weight:700;color:{fill_color};width:28px;text-align:right;">{value:.2f}</div>'
                f'</div>'
            )

        chips_html = "".join(
            f'<span style="font-size:10px;background:#EEEDFE;color:#534AB7;'
            f'padding:3px 9px;border-radius:20px;margin:2px 3px 2px 0;display:inline-block;">'
            f'{sol["label"]}</span>'
            for sol in opp.get("solutions", [])
        )

        card_html = (
            f'<div style="border:1.5px solid #E0DFDB;{top_border}border-radius:10px;'
            f'background:#FFFFFF;padding:16px;position:relative;height:100%;">'
            f'<div style="position:absolute;top:12px;right:12px;font-size:9px;font-weight:700;'
            f'background:{rank_bg};color:{rank_color};padding:2px 8px;border-radius:20px;">{rank_label}</div>'
            f'<div style="font-size:12px;font-weight:600;color:#26215C;line-height:1.45;'
            f'margin-bottom:14px;padding-right:70px;">{opp["jtbd"]}</div>'
            + score_bar("Priority score (final ranking)", ps, "#E24B4A")
            + score_bar("ODI score (unmet need)", odi, "#534AB7")
            + score_bar("Evidence robustness (source diversity)", rob, "#0C447C")
            + f'<div style="font-size:10px;font-weight:700;color:#534AB7;text-transform:uppercase;'
            f'letter-spacing:0.06em;margin:12px 0 6px;">Proposed solutions</div>'
            + chips_html
            + f'</div>'
        )

        with col:
            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)
            if st.button("View evidence", key=f"opp_btn_{idx}"):
                st.session_state["selected_opp"] = idx
                st.rerun()

    # ── Evidence trail ────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    sel = st.session_state.get("selected_opp", 0)
    if sel < len(opportunities):
        selected_opp = opportunities[sel]
        selected_cid = selected_opp.get("cluster_id")

        evidence_chunks = [
            (cid, ch) for cid, ch in source_map.items()
            if str(ch.get("cluster_id", "")) == str(selected_cid)
        ]
        ev_types = sorted({ch.get("source_type", "") for _, ch in evidence_chunks if ch.get("source_type")})

        badges_html = "".join(
            f'<span style="font-size:11px;color:#534AB7;font-weight:500;margin-right:4px;">'
            f'{"· " if i else ""}{t}</span>'
            for i, t in enumerate(ev_types)
        )

        q_word  = "quotes" if len(evidence_chunks) != 1 else "quote"
        t_word  = "source types" if len(ev_types) != 1 else "source type"
        ev_note = f" — {len(evidence_chunks)} {q_word} across {len(ev_types)} {t_word}" if evidence_chunks else ""
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:10px;">'
            f'Source Traceability · Opportunity #{sel+1}{ev_note}'
            f'</div>',
            unsafe_allow_html=True,
        )

        if evidence_chunks:
            st.markdown(
                f'<div style="background:#FFFFFF;border:1.5px solid #E0DFDB;border-radius:10px;padding:16px;">'
                f'<div style="font-size:13px;font-weight:700;color:#1A1916;margin-bottom:12px;display:flex;align-items:center;gap:10px;">'
                f'Evidence trail '
                f'<span style="background:#EEEDFE;color:#534AB7;font-size:10px;font-weight:700;padding:2px 10px;border-radius:20px;">'
                f'{" · ".join(ev_types)}</span></div></div>',
                unsafe_allow_html=True,
            )
            ev_cols = st.columns(min(len(evidence_chunks), 3))
            for i, (cid, ch) in enumerate(evidence_chunks[:3]):
                src  = ch.get("source_type", "")
                dot  = SOURCE_DOT.get(src, "#888888")
                fname = ch.get("filename", "")
                with ev_cols[i]:
                    st.markdown(
                        f'<div style="background:#FAFAF8;border:1px solid #E0DFDB;border-radius:8px;padding:10px 12px;">'
                        f'<div style="font-size:9px;font-weight:700;color:#888;text-transform:uppercase;'
                        f'letter-spacing:0.07em;margin-bottom:5px;display:flex;align-items:center;gap:5px;">'
                        f'<span style="width:7px;height:7px;border-radius:50%;background:{dot};display:inline-block;"></span>'
                        f'{src} · {fname}</div>'
                        f'<div style="font-size:11px;color:#1A1916;line-height:1.5;font-style:italic;">'
                        f'"{ch.get("text","")[:240]}{"…" if len(ch.get("text",""))>240 else ""}"</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No evidence chunks linked to this opportunity.")

# ── TAB 2: PRIORITY MATRIX ───────────────────────────────────────────────────
with tab_matrix:
    if not opportunities:
        st.info("Run the pipeline to see the priority matrix.")
    else:
        col_chart, col_legend = st.columns([3, 2])
        with col_chart:
            fig = go.Figure()

            quadrants = [
                (0, 0.5, 0.5, 1, "#EEEDFE", "watch"),
                (0.5, 1, 0.5, 1, "#F7C1C1", "prioritise"),
                (0, 0.5, 0, 0.5, "#F1EFE8", "deprioritise"),
                (0.5, 1, 0, 0.5, "#EAF3DE", "maintain"),
            ]
            for x0, x1, y0, y1, color, label in quadrants:
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                              fillcolor=color, opacity=0.5, line_width=0)
                fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=label,
                                   showarrow=False, font=dict(size=11, color="#888"))

            for i, opp in enumerate(opportunities):
                score = opp.get("priority_score") or 0
                fig.add_trace(go.Scatter(
                    x=[opp.get("importance", 0)],
                    y=[opp.get("satisfaction", 0)],
                    mode="markers+text",
                    marker=dict(size=max(score * 60, 20), color="#534AB7",
                                opacity=0.4 + score * 0.6),
                    text=[str(i + 1)],
                    textposition="middle center",
                    textfont=dict(color="white", size=12),
                    hovertemplate=(
                        f"<b>Opp {i+1}</b><br>"
                        f"Importance: {opp.get('importance',0):.2f}<br>"
                        f"Satisfaction: {opp.get('satisfaction',0):.2f}<br>"
                        f"ODI score: {opp.get('odi_score',0):.2f}<br>"
                        f"Robustness: {opp.get('evidence_robustness',0):.2f}<br>"
                        f"Priority score: {score:.2f}<extra></extra>"
                    ),
                    showlegend=False,
                ))

            fig.update_layout(
                xaxis=dict(title="Importance →", range=[0, 1], showgrid=False),
                yaxis=dict(title="Satisfaction →", range=[0, 1], showgrid=False),
                plot_bgcolor="white",
                margin=dict(l=40, r=20, t=20, b=40),
                height=360,
            )
            fig.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1,
                          line=dict(color="#D3D1C7", width=1))
            fig.add_shape(type="line", x0=0, x1=1, y0=0.5, y1=0.5,
                          line=dict(color="#D3D1C7", width=1))
            st.plotly_chart(fig, use_container_width=True)

        with col_legend:
            st.markdown("**Opportunities by priority score**")
            for i, opp in enumerate(opportunities):
                score = opp.get("priority_score") or 0
                short = opp["jtbd"].split(",")[0].replace("When I ", "").capitalize()[:40]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
                    f'<div style="width:10px;height:10px;border-radius:50%;background:#534AB7;'
                    f'opacity:{0.4+score*0.6};flex-shrink:0;"></div>'
                    f'<div style="font-size:12px;"><b>{i+1}</b> — {short}… ({score:.2f})</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                "<div style='margin-top:16px;font-size:11px;color:#888;line-height:1.7;'>"
                "Bubble size = priority score<br>"
                "X axis = importance (cluster size / total chunks)<br>"
                "Y axis = satisfaction ((sentiment+1)/2)<br>"
                "<b>Top-right = prioritise</b> · <b>Bottom-right = maintain</b>"
                "</div>",
                unsafe_allow_html=True,
            )

# ── TAB 3: EVIDENCE HEATMAP ───────────────────────────────────────────────────
with tab_heatmap:
    if not source_map:
        st.info("Run the pipeline to see the evidence heatmap.")
    else:
        st.caption("Chunk count per opportunity × source type. Wider bar = more evidence.")
        source_types = sorted({c.get("source_type", "") for c in source_map.values() if c.get("source_type")})
        bar_colors = ["#CECBF6", "#B5D4F4", "#9FE1CB", "#D3D1C7", "#F4C4A0", "#C4E0C4"]

        for i, opp in enumerate(opportunities):
            cid = opp.get("cluster_id")
            short = opp["jtbd"].split(",")[0][:55]
            st.markdown(f"**Opp {i+1} — {short}…**")

            chunks_for_opp = [c for c in source_map.values() if str(c.get("cluster_id", "")) == str(cid)]
            total = len(chunks_for_opp) or 1
            has_any = False
            for src in source_types:
                count = sum(1 for c in chunks_for_opp if c.get("source_type") == src)
                if count == 0:
                    continue
                has_any = True
                bar_pct = int((count / max(total, 1)) * 260)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                    f'<div style="font-size:12px;color:#888;width:120px;">{src.capitalize()}</div>'
                    f'<div style="height:20px;width:{max(bar_pct,30)}px;border-radius:3px;'
                    f'background:{bar_colors[i % len(bar_colors)]};display:flex;align-items:center;'
                    f'padding-left:6px;font-size:11px;color:#26215C;font-weight:500;">'
                    f'{count} chunk{"s" if count != 1 else ""}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if not has_any:
                st.caption("No evidence chunks linked yet.")
            st.markdown("---")
