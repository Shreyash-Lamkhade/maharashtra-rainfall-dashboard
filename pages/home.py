import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_main, load_monthly

df  = load_main()
mon = load_monthly()

st.title("🌧️ Maharashtra Rainfall — Data Analytics Dashboard")
st.markdown("A complete DA project on Maharashtra daily rainfall data (2021–2025), covering preprocessing, regression, association mining, and decision trees.")

st.markdown("---")

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records",   f"{len(df):,}")
c2.metric("Districts",       df["District"].nunique())
c3.metric("Stations",        df["Station"].nunique())
c4.metric("Years",           f"{df['Year'].min()}–{df['Year'].max()}")

st.markdown("---")

# Annual rainfall bar chart
st.subheader("Annual Total Rainfall (mm)")
annual = df.groupby("Year")["Rainfall_mm"].sum().reset_index()
fig = px.bar(annual, x="Year", y="Rainfall_mm",
             color="Rainfall_mm", color_continuous_scale="Blues",
             text_auto=".2s", labels={"Rainfall_mm": "Total Rainfall (mm)"})
fig.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

# Monthly heatmap
st.subheader("District × Month Rainfall Heatmap")
pivot = (
    mon.groupby(["District","Month"])["Total_Rainfall"]
    .mean().reset_index()
    .pivot(index="District", columns="Month", values="Total_Rainfall")
    .fillna(0)
)
mnames = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
pivot.columns = [mnames.get(c,c) for c in pivot.columns]
fig2 = px.imshow(pivot, color_continuous_scale="Blues", aspect="auto",
                 labels={"color":"Avg Rainfall (mm)"})
fig2.update_layout(height=550)
st.plotly_chart(fig2, use_container_width=True)

# Top 10 districts
st.subheader("Top 10 Rainiest Districts")
top = df.groupby("District")["Rainfall_mm"].sum().nlargest(10).reset_index()
fig3 = px.bar(top, x="Rainfall_mm", y="District", orientation="h",
              color="Rainfall_mm", color_continuous_scale="Teal",
              labels={"Rainfall_mm":"Total Rainfall (mm)"})
fig3.update_layout(yaxis={"categoryorder":"total ascending"},
                   coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.info("Use the sidebar to navigate through each module: Preprocessing → Regression → Association Mining → Decision Tree.")
