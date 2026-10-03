"""HYDRA-RAG dashboard entry point.

Run from the repo root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Add both repo root and app directory to sys.path
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for p in (str(REPO_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

# Direct local imports bypassing the colliding "app" namespace
try:
    from app.ui_helpers import get_index_size, inject_css
    from app.views import page_dashboard, page_search, warmup_retrieval_pipeline
except ModuleNotFoundError:
    from ui_helpers import get_index_size, inject_css
    from views import page_dashboard, page_search, warmup_retrieval_pipeline

st.set_page_config(page_title="HYDRA-RAG", page_icon=":material/hub:", layout="wide")
inject_css()

# Pre-warm runtime tensors, tokenizers, and DB socket connections
warmup_retrieval_pipeline()

page = st.navigation([
    st.Page(page_search, title="Search", icon=":material/search:", default=True),
    st.Page(page_dashboard, title="Dashboard", icon=":material/dashboard:"),
])
size = get_index_size()
st.sidebar.caption(f"Indexed passages: {size:,}" if size else "Qdrant not reachable")
page.run()