import streamlit as st

st.set_page_config(
    page_title="Maharashtra Rainfall — DA Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("🌧️ DA Project")
st.sidebar.caption("Maharashtra Rainfall 2021–2025")
st.sidebar.markdown("---")

PAGES = [
    "🏠 Home",
    "🧹 Data Preprocessing",
    "📈 Regression Analysis",
    "🔗 Association Mining",
    "🌲 Decision Tree (ID3/C4.5/CART)",
]
page = st.sidebar.radio("Navigate to", PAGES)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** Maharashtra Daily Rainfall  \n"
    "**Records:** 1,21,292  \n"
    "**Districts:** 33  \n"
    "**Period:** Jan 2021 – 2025"
)

# ── Route ────────────────────────────────────────────────────────────────────
import importlib.util, os

PAGE_FILES = {
    "🏠 Home":                        "pages/home.py",
    "🧹 Data Preprocessing":          "pages/preprocessing.py",
    "📈 Regression Analysis":         "pages/regression.py",
    "🔗 Association Mining":           "pages/association.py",
    "🌲 Decision Tree (ID3/C4.5/CART)":"pages/decision_tree.py",
}

page_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PAGE_FILES[page])
spec = importlib.util.spec_from_file_location("_page", page_path)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
