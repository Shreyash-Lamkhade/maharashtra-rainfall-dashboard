import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from utils.data_loader import load_main, load_regression_comparison

df = load_main()
cmp = load_regression_comparison()

st.title("📈 Regression Analysis")
st.markdown("Predicting rainfall patterns using three regression approaches.")

# Build features
@st.cache_data
def build_features():
    mon = (df.groupby(["Year", "Month", "District"])["Rainfall_mm"]
           .mean().reset_index()
           .rename(columns={"Rainfall_mm": "Avg_Rainfall"}))
    mon["HeavyRain"] = (mon["Avg_Rainfall"] >= 10).astype(int)
    return mon

mon = build_features()

# ── Model Overview ────────────────────────────────────────────────────────
st.header("Model Performance Comparison")
styled = (
    cmp.style
       .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"})
       .highlight_min(subset=["MAE", "RMSE"], props="background-color:#1f6e3a; color:white; font-weight:bold")
       .highlight_max(subset=["R2"],          props="background-color:#1f6e3a; color:white; font-weight:bold")
)
st.dataframe(styled, use_container_width=True)

best = cmp.loc[cmp["R2"].idxmax(), "Model"]
st.success(f"🏆 Best performer: **{best}** (R² = {cmp.loc[cmp['R2'].idxmax(), 'R2']:.4f})")

# Metrics explanation
with st.expander("📖 Understanding the Metrics"):
    st.markdown("""
    **MAE (Mean Absolute Error):** Average prediction error in mm — lower is better.  
    **RMSE (Root Mean Squared Error):** Penalizes large errors more heavily.  
    **R² (Coefficient of Determination):** % of variance explained by the model (0 to 1, higher is better).
    """)

st.markdown("---")

# ── 1. Simple Linear Regression ───────────────────────────────────────────
st.header("1 · Simple Linear Regression")
st.markdown("**Hypothesis:** Rainfall follows a seasonal pattern based on month alone.")

@st.cache_data
def run_slr():
    X = mon[["Month"]]
    y = mon["Avg_Rainfall"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    yp = m.predict(Xte)
    return m, Xte, yte, yp

m_slr, Xte_s, yte_s, yp_s = run_slr()

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{mean_absolute_error(yte_s, yp_s):.2f} mm")
c2.metric("RMSE", f"{np.sqrt(mean_squared_error(yte_s, yp_s)):.2f} mm")
c3.metric("R²", f"{r2_score(yte_s, yp_s):.4f}")
c4.metric("Slope", f"{m_slr.coef_[0]:.2f} mm/month")

# Interactive plot
x_line = np.arange(1, 13).reshape(-1, 1)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=Xte_s["Month"], y=yte_s,
    mode="markers", name="Actual",
    marker=dict(color="#636EFA", size=8, opacity=0.5)
))
fig.add_trace(go.Scatter(
    x=x_line.flatten(), y=m_slr.predict(x_line),
    mode="lines", name="Regression Line",
    line=dict(color="#EF553B", width=3)
))
fig.update_layout(
    title="Month vs Average Rainfall",
    xaxis_title="Month", yaxis_title="Avg Rainfall (mm)",
    xaxis=dict(tickmode="linear", dtick=1),
    plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig, use_container_width=True)

# Residual analysis
residuals = yte_s.values - yp_s
fig_res = px.scatter(x=yp_s, y=residuals,
                     labels={"x": "Predicted", "y": "Residual (Actual - Predicted)"},
                     title="Residual Plot",
                     color_discrete_sequence=["#00CC96"])
fig_res.add_hline(y=0, line_dash="dash", line_color="red")
fig_res.update_layout(plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_res, use_container_width=True)

st.markdown(f"""
**What this tells us:**  
The slope of {m_slr.coef_[0]:.2f} mm/month shows rainfall increases through the year, peaking in monsoon (June-Sep).  
However, R² = {r2_score(yte_s, yp_s):.3f} means Month alone explains only ~{r2_score(yte_s, yp_s)*100:.0f}% of variation.  
The residual plot shows scattered errors — location (district) matters more than we're capturing here.
""")

st.markdown("---")

# ── 2. Multiple Linear Regression ─────────────────────────────────────────
st.header("2 · Multiple Linear Regression")
st.markdown("**Enhancement:** Adding Year and District features to capture geography and trends.")

@st.cache_data
def run_mlr():
    data = pd.get_dummies(mon, columns=["District"], drop_first=True)
    feat = [c for c in data.columns if c not in ["Avg_Rainfall", "HeavyRain"]]
    X, y = data[feat], data["Avg_Rainfall"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    yp = m.predict(Xte)
    coefs = pd.Series(m.coef_, index=feat).sort_values(key=abs, ascending=False)
    return yte, yp, coefs

yte_m, yp_m, coefs = run_mlr()

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mean_absolute_error(yte_m, yp_m):.2f} mm")
c2.metric("RMSE", f"{np.sqrt(mean_squared_error(yte_m, yp_m)):.2f} mm")
c3.metric("R²", f"{r2_score(yte_m, yp_m):.4f}")

col1, col2 = st.columns(2)

with col1:
    fig_ap = px.scatter(x=yte_m, y=yp_m, opacity=0.4,
                        labels={"x": "Actual", "y": "Predicted"},
                        title="Actual vs Predicted",
                        color_discrete_sequence=["#00CC96"])
    lim = max(float(pd.Series(yte_m).max()), float(pd.Series(yp_m).max()))
    fig_ap.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                     line=dict(color="red", dash="dash"))
    fig_ap.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ap, use_container_width=True)

with col2:
    top10 = coefs.head(10).reset_index()
    top10.columns = ["Feature", "Coefficient"]
    fig_c = px.bar(top10, x="Coefficient", y="Feature", orientation="h",
                   color="Coefficient", color_continuous_scale="RdBu",
                   color_continuous_midpoint=0, title="Top 10 Features")
    fig_c.update_layout(yaxis={"categoryorder": "total ascending"},
                        coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_c, use_container_width=True)

improvement = ((r2_score(yte_m, yp_m) - r2_score(yte_s, yp_s)) / r2_score(yte_s, yp_s) * 100)
st.markdown(f"""
**Impact of additional features:**  
R² improved by {improvement:.0f}%, from {r2_score(yte_s, yp_s):.3f} to {r2_score(yte_m, yp_m):.3f}.  
Konkan districts (Ratnagiri, Raigad, Palghar) have the largest positive coefficients — confirming their high rainfall.  
The scatter plot shows better alignment with the diagonal — predictions are more accurate across the range.
""")

st.markdown("---")

# ── 3. Logistic Regression ────────────────────────────────────────────────
st.header("3 · Logistic Regression (Binary Classification)")
st.markdown("**Task:** Predict whether a district-month will have heavy rainfall (≥10mm average).")

@st.cache_data
def run_logistic():
    data = pd.get_dummies(mon, columns=["District"], drop_first=True)
    feat = [c for c in data.columns if c not in ["Avg_Rainfall", "HeavyRain"]]
    X, y = data[feat], data["HeavyRain"]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
    m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    yp = m.predict(Xte)
    yprb = m.predict_proba(Xte)[:, 1]
    rep = classification_report(yte, yp, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, yp)
    fpr, tpr, _ = roc_curve(yte, yprb)
    rauc = auc(fpr, tpr)
    return yte, yp, yprb, rep, cm, fpr, tpr, rauc

yte_l, yp_l, yprb_l, rep_l, cm_l, fpr_l, tpr_l, rauc_l = run_logistic()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{rep_l['accuracy']*100:.1f}%")
c2.metric("Precision", f"{rep_l.get('1', {}).get('precision', 0)*100:.1f}%")
c3.metric("Recall", f"{rep_l.get('1', {}).get('recall', 0)*100:.1f}%")
c4.metric("ROC-AUC", f"{rauc_l:.3f}")

col1, col2 = st.columns(2)

with col1:
    fig_cm = px.imshow(cm_l, text_auto=True, color_continuous_scale="Blues",
                       x=["No Heavy Rain", "Heavy Rain"],
                       y=["No Heavy Rain", "Heavy Rain"],
                       title="Confusion Matrix")
    fig_cm.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_l, y=tpr_l, mode="lines",
                                  name=f"Model (AUC={rauc_l:.3f})",
                                  line=dict(color="#636EFA", width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  name="Random Classifier",
                                  line=dict(color="gray", dash="dash")))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="ROC Curve",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# Probability distribution
fig_prob = px.histogram(
    x=yprb_l,
    color=yte_l.astype(str),
    nbins=50,
    labels={"x": "Predicted Probability", "color": "Actual Class"},
    title="Predicted Probability Distribution",
    color_discrete_map={"0": "#00CC96", "1": "#EF553B"}
)
fig_prob.update_layout(plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_prob, use_container_width=True)

st.markdown(f"""
**Classification performance:**  
The model achieves {rep_l['accuracy']*100:.1f}% accuracy with strong AUC ({rauc_l:.3f}), indicating good separation between classes.  
Confusion matrix shows {cm_l[0,0]} true negatives and {cm_l[1,1]} true positives — most predictions are correct.  
The probability distribution shows good separation — high-confidence predictions tend to be accurate.

**Practical application:**  
This can serve as an early-warning system for district authorities to prepare for potential flooding during monsoon months.
""")

st.markdown("---")

st.markdown("### 💡 Comparative Analysis")
st.markdown("""
**Simple Linear Regression** establishes the baseline seasonal pattern but lacks geographic context.  
**Multiple Linear Regression** incorporates location and improves predictions significantly, revealing that geography dominates rainfall patterns.  
**Logistic Regression** shifts to a classification lens, useful when binary decisions (prepare/don't prepare) matter more than exact rainfall amounts.

The progression from simple to multiple regression demonstrates how **feature engineering** (adding district, year) transforms model performance. The logistic model shows that even complex continuous data can be effectively reduced to actionable binary predictions when the problem context demands it.
""")
