"""
Discovery Lens — Entry point
Initialises session state keys and renders the home page.
"""
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

st.set_page_config(
    page_title="Discovery Lens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

defaults = {
    "goal": "",
    "product_name": "",
    "goal_validated": False,
    "context_block": "",
    "chunks": [],
    "embeddings": None,
    "clusters": [],
    "scored_clusters": [],
    "ost": {},
    "source_map": {},
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("🔍 Discovery Lens")
st.subheader("Turn qualitative discovery data into a structured Opportunity-Solution Tree")
st.markdown("""
**How it works:**
1. **Set your goal** — Enter your product name and goal statement
2. **Upload your files** — Interviews, reviews, support tickets, usability notes (PDF, DOCX, CSV, TXT)
3. **Get your OST** — The tool clusters insights, frames opportunities in JTBD language, and shows you the evidence behind every decision

👈 Use the sidebar to get started.
""")

if st.checkbox("Show session state (debug)", value=False):
    st.json({k: str(v)[:200] if v else v for k, v in st.session_state.items()})
