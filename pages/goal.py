import streamlit as st
from pipeline.goal_validator import validate_goal

st.set_page_config(page_title="Discovery Lens", layout="centered")

# ── SESSION STATE ──────────────────────────────────────────────────────────────
for key, default in [
    ("product_name", ""),
    ("goal", ""),
    ("current_screen", 1),
    ("goal_validated", False),
    ("_val_result", None),
    ("_val_goal", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── GOAL BANNER ────────────────────────────────────────────────────────────────
_goal_preview = st.session_state.get("goal", "")
if _goal_preview:
    st.markdown(
        f'<div style="background:#EEEDFE;border-left:4px solid #534AB7;border-radius:6px;'
        f'padding:10px 16px;font-size:13px;color:#534AB7;margin-bottom:16px;">'
        f'<span style="font-weight:600;">Goal:</span> {_goal_preview}</div>',
        unsafe_allow_html=True,
    )

# ── STEPPER ────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**① Goal** ← you are here")
with col2:
    st.markdown("② Upload")
with col3:
    st.markdown("③ Results")
st.markdown("---")

# ── FORM ───────────────────────────────────────────────────────────────────────
st.title("Define your product goal")
st.caption("This goal will be passed into every AI prompt to frame the analysis.")
st.markdown("<small style='color: gray;'>* required field</small>", unsafe_allow_html=True)

product_name = st.text_input(
    label="Product name *",
    key="_product_name_input",
    value=st.session_state.get("product_name", ""),
    placeholder="e.g. Asana",
)

goal_statement = st.text_area(
    label="Goal statement *",
    key="_goal_input",
    value=st.session_state.get("goal", ""),
    placeholder=(
        "e.g. Increase the percentage of new teams that complete "
        "their first project within 30 days of signing up"
    ),
    height=120,
)

# ── NEXT BUTTON ────────────────────────────────────────────────────────────────
if st.button("Next: upload files →", use_container_width=True, type="primary"):

    if not product_name.strip():
        st.error("Please enter a product name before continuing.")

    elif not goal_statement.strip():
        st.error("Please enter a goal statement before continuing.")

    else:
        st.session_state["product_name"] = product_name.strip()
        st.session_state["goal"] = goal_statement.strip()

        # Re-validate only if goal text changed since last check
        if st.session_state["_val_goal"] != goal_statement.strip():
            st.session_state["_val_result"] = None

        if st.session_state["_val_result"] is None:
            with st.spinner("Checking goal quality…"):
                result = validate_goal(goal_statement.strip())
            st.session_state["_val_result"] = result
            st.session_state["_val_goal"] = goal_statement.strip()

        result = st.session_state["_val_result"]

        if result["passed"]:
            st.session_state["goal_validated"] = True
            st.session_state["_val_result"] = None
            st.session_state["current_screen"] = 2
            st.switch_page("pages/upload.py")
        else:
            st.rerun()

# ── VALIDATION FEEDBACK ────────────────────────────────────────────────────────
val = st.session_state.get("_val_result")
if val is not None and not val["passed"]:

    st.markdown("---")
    st.warning(
        f"Your goal scored **{val['score']}/3** on goal quality. "
        "Strong goals have a measurable metric, a named user segment, and a timeframe. "
        "You can edit your goal above, use the suggested rewrite, or proceed anyway."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"{'✅' if val['measurable_metric'] else '❌'} **Measurable metric**")
    with c2:
        st.markdown(f"{'✅' if val['user_segment'] else '❌'} **User segment**")
    with c3:
        st.markdown(f"{'✅' if val['timeframe'] else '❌'} **Timeframe**")

    if val["feedback"]:
        st.caption(val["feedback"])

    if val.get("rewrite"):
        st.info(f"💡 **Suggested rewrite:** {val['rewrite']}")

        col_accept, col_override = st.columns(2)
        with col_accept:
            if st.button("Use suggested rewrite →", type="primary", use_container_width=True):
                st.session_state["goal"] = val["rewrite"]
                st.session_state["goal_validated"] = True
                st.session_state["_val_result"] = None
                st.session_state["current_screen"] = 2
                st.switch_page("pages/upload.py")
        with col_override:
            if st.button("Proceed with my goal anyway", use_container_width=True):
                st.session_state["goal_validated"] = True
                st.session_state["_val_result"] = None
                st.session_state["current_screen"] = 2
                st.switch_page("pages/upload.py")
    else:
        if st.button("Proceed anyway", use_container_width=True):
            st.session_state["goal_validated"] = True
            st.session_state["_val_result"] = None
            st.session_state["current_screen"] = 2
            st.switch_page("pages/upload.py")
