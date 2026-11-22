import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_main

df = load_main()

st.title("🧹 Data Preprocessing")
st.markdown("Cleaning and transforming raw data into analysis-ready format.")

# ── 1. Dataset Overview ──────────────────────────────────────────────────────
st.header("1 · Raw Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Missing Values", int(df.isna().sum().sum()))

with st.expander("Show sample data"):
    st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# ── 2. Data Quality Analysis ────────────────────────────────────────────────
st.header("2 · Data Quality Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Duplicate Records")
    dup_data = pd.DataFrame({
        "Stage": ["Raw Dataset", "After Deduplication"],
        "Records": [121904, 121292],
    })
    fig_dup = px.bar(dup_data, x="Stage", y="Records", text="Records",
                     color="Stage", color_discrete_sequence=["#EF553B","#00CC96"])
    fig_dup.update_traces(textposition="outside")
    fig_dup.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_dup, use_container_width=True)
    st.caption("612 duplicates removed (same Station + Timestamp)")

with col2:
    st.subheader("Data Completeness")
    missing = df.isnull().sum()
    missing = missing[missing > 0].reset_index()
    if len(missing) == 0:
        st.success("✅ No missing values detected")
    else:
        missing.columns = ["Column", "Missing"]
        missing["% Missing"] = (missing["Missing"] / len(df) * 100).round(2)
        fig_miss = px.bar(missing, x="Column", y="% Missing",
                          color="% Missing", color_continuous_scale="Reds")
        st.plotly_chart(fig_miss, use_container_width=True)

st.markdown("---")

# ── 3. Outlier Detection (Interactive) ─────────────────────────────────────
st.header("3 · Outlier Detection Using IQR Method")

col1, col2 = st.columns([2, 1])
with col2:
    st.markdown("**IQR Formula:**")
    st.latex(r"IQR = Q_3 - Q_1")
    st.latex(r"Lower = Q_1 - 1.5 \times IQR")
    st.latex(r"Upper = Q_3 + 1.5 \times IQR")

district = st.selectbox("Select district to analyze", sorted(df["District"].dropna().unique()))

@st.cache_data
def analyze_outliers(dist):
    sub = df[df["District"] == dist]["Rainfall_mm"].dropna()
    Q1, Q3 = sub.quantile(0.25), sub.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = sub[(sub < lo) | (sub > hi)]
    return sub, Q1, Q3, IQR, lo, hi, outliers

sub, Q1, Q3, IQR, lo, hi, outliers = analyze_outliers(district)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Q1 (25th %ile)", f"{Q1:.1f} mm")
c2.metric("Q3 (75th %ile)", f"{Q3:.1f} mm")
c3.metric("IQR", f"{IQR:.1f} mm")
c4.metric("Outliers", f"{len(outliers)} ({len(outliers)/len(sub)*100:.1f}%)")

# Box plot + scatter overlay
fig = go.Figure()
fig.add_trace(go.Box(
    y=sub,
    name=district,
    boxpoints="outliers",
    marker_color="#636EFA",
    line_color="#636EFA"
))
fig.update_layout(
    title=f"Rainfall Distribution — {district}",
    yaxis_title="Rainfall (mm)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Time series view of outliers
all_dist = df[df["District"] == district][["Date", "Rainfall_mm"]].dropna().copy()
all_dist["Type"] = np.where(
    (all_dist["Rainfall_mm"] < lo) | (all_dist["Rainfall_mm"] > hi),
    "Outlier", "Normal"
)
fig2 = px.scatter(all_dist, x="Date", y="Rainfall_mm", color="Type",
                  color_discrete_map={"Normal": "#00CC96", "Outlier": "#EF553B"},
                  opacity=0.6)
fig2.add_hline(y=hi, line_dash="dash", line_color="orange",
               annotation_text=f"Upper fence: {hi:.1f} mm")
fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=350)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── 4. Normalization Techniques ───────────────────────────────────────────
st.header("4 · Normalization & Standardization")

# Select a district for demo
raw = df[df["District"] == district]["Rainfall_mm"].dropna().values

if len(raw) > 0:
    # Calculate normalized versions
    mm_norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    z_norm = (raw - raw.mean()) / (raw.std() + 1e-8)
    
    tab1, tab2, tab3 = st.tabs(["📊 Compare All", "🔢 Min-Max", "📈 Z-Score"])
    
    with tab1:
        st.subheader("Comparison of Normalization Methods")
        comparison_df = pd.DataFrame({
            "Original": raw[:500],
            "Min-Max [0,1]": mm_norm[:500],
            "Z-Score (μ=0, σ=1)": z_norm[:500]
        })
        
        fig_comp = go.Figure()
        for col in comparison_df.columns:
            fig_comp.add_trace(go.Histogram(
                x=comparison_df[col],
                name=col,
                opacity=0.6,
                nbinsx=40
            ))
        fig_comp.update_layout(
            barmode='overlay',
            title="Distribution Comparison",
            xaxis_title="Value",
            yaxis_title="Frequency",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_comp, use_container_width=True, key="norm_compare")
        
        # Statistics table
        stats = pd.DataFrame({
            "Metric": ["Min", "Max", "Mean", "Std Dev"],
            "Original": [raw.min(), raw.max(), raw.mean(), raw.std()],
            "Min-Max": [mm_norm.min(), mm_norm.max(), mm_norm.mean(), mm_norm.std()],
            "Z-Score": [z_norm.min(), z_norm.max(), z_norm.mean(), z_norm.std()]
        }).round(3)
        st.dataframe(stats, use_container_width=True)
    
    with tab2:
        st.markdown("**Min-Max Normalization** scales data to [0, 1] range:")
        st.latex(r"x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}")
        st.markdown("**Use case:** Distance-based algorithms (KNN, K-Means)")
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.histogram(raw, nbins=50, title="Before",
                        labels={"value": "Rainfall (mm)"},
                        color_discrete_sequence=["#636EFA"]),
            use_container_width=True,
            key="mm_before"
        )
        c2.plotly_chart(
            px.histogram(mm_norm, nbins=50, title="After Min-Max",
                        labels={"value": "Normalized [0-1]"},
                        color_discrete_sequence=["#00CC96"]),
            use_container_width=True,
            key="mm_after"
        )
    
    with tab3:
        st.markdown("**Z-Score Standardization** centers data around mean=0, std=1:")
        st.latex(r"z = \frac{x - \mu}{\sigma}")
        st.markdown("**Use case:** Linear regression, SVM, neural networks")
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.histogram(raw, nbins=50, title="Before",
                        labels={"value": "Rainfall (mm)"},
                        color_discrete_sequence=["#636EFA"]),
            use_container_width=True,
            key="zs_before"
        )
        c2.plotly_chart(
            px.histogram(z_norm, nbins=50, title="After Z-Score",
                        labels={"value": "z-score"},
                        color_discrete_sequence=["#EF553B"]),
            use_container_width=True,
            key="zs_after"
        )

st.markdown("---")

# ── Interpretation ────────────────────────────────────────────────────────
st.markdown("### 💡 Key Insights")
st.markdown(f"""
**Data Quality:**  
Removing 612 duplicates ensures each rainfall measurement is counted only once, preventing bias in aggregate statistics.

**Outlier Analysis:**  
The IQR method flags extreme rainfall events (cyclones, cloudbursts) which are genuine but rare. For {district}, 
{len(outliers)/len(sub)*100:.1f}% of readings are outliers — these need special handling in predictive models to avoid skewing results.

**Normalization Strategy:**  
- **Min-Max** preserves the original distribution shape and is ideal when features need equal weight in distance calculations.
- **Z-Score** handles outliers better and is preferred for regression models where we care about how far a value deviates from the mean.

For Maharashtra's monsoon data, Z-score normalization works better because it doesn't let extreme Konkan rainfall dominate the scale.
""")
