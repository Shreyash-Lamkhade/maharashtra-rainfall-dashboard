"""Shared data-loading utilities — cached so each page doesn't reload from disk."""
import os
import pandas as pd
import streamlit as st

# Works both locally (CSV in parent folder) and on Streamlit Cloud (CSV in same folder)
_HERE = os.path.dirname(os.path.abspath(__file__))

def _find(filename):
    """Search in utils/../ and utils/../../ so the app works wherever CSVs are placed."""
    for folder in [
        os.path.join(_HERE, ".."),           # dashboard/
        os.path.join(_HERE, "..", ".."),     # Weather DA/
        _HERE,                               # utils/ (fallback)
    ]:
        p = os.path.normpath(os.path.join(folder, filename))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{filename} not found. Place it in the dashboard folder.")

@st.cache_data(show_spinner="Loading dataset…")
def load_main() -> pd.DataFrame:
    return pd.read_csv(_find("maharashtra_rainfall_clean_rebuilt.csv"), parse_dates=["Date"])

@st.cache_data(show_spinner="Loading monthly summary…")
def load_monthly() -> pd.DataFrame:
    return pd.read_csv(_find("monthly_rainfall_analysis.csv"))

@st.cache_data(show_spinner="Loading regression comparison…")
def load_regression_comparison() -> pd.DataFrame:
    return pd.read_csv(_find("regression_comparison.csv"))
