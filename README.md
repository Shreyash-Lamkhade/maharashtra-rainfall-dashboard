# Maharashtra Rainfall — Data Analytics Dashboard

A comprehensive interactive dashboard demonstrating data analytics concepts on Maharashtra rainfall data (2021-2025).

## 🚀 Quick Start

### Option 1: Double-click to run
1. Open `C:\Sem 5\EDA\Weather DA\dashboard\` folder
2. Double-click **`setup_and_run.bat`**
3. Dashboard opens at **http://localhost:8501**

### Option 2: Manual setup
```bash
cd "C:\Sem 5\EDA\Weather DA\dashboard"
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Dashboard Modules

### 🏠 Home
- Dataset overview with KPIs
- Annual rainfall trends
- District-wise heatmaps
- Top 10 rainiest districts

### 🧹 Data Preprocessing
- Data quality analysis (duplicates, missing values)
- Interactive outlier detection (IQR method)
- Normalization comparison (Min-Max vs Z-Score)
- Time-series outlier visualization

### 📈 Regression Analysis
- **Simple Linear Regression** — seasonal pattern
- **Multiple Linear Regression** — with geography
- **Logistic Regression** — heavy rainfall classification
- Residual analysis, feature importance, ROC curves

### 🔗 Association Rule Mining
- **Sampling & Partitioning** strategies
- **Apriori** algorithm with anti-monotone pruning
- **FP-Growth** (compressed FP-Tree)
- **Eclat** (vertical data format)
- **Dynamic Counting** — support convergence
- **PCY Hash-Based** technique
- District co-occurrence analysis

### 🌲 Decision Tree Classification
- **ID3** (Information Gain)
- **C4.5** (Gain Ratio)
- **CART** (Gini Impurity)
- Confusion matrices, feature importance
- Depth vs accuracy trade-off
- Tree visualization

## 🎯 Key Features

- **Interactive Controls** — sliders, selectors for real-time parameter tuning
- **Visual Comparisons** — side-by-side algorithm performance
- **Deep Insights** — not just results, but interpretation and understanding
- **Clean Design** — no clutter, focused on the analysis

## 📦 Requirements

- Python 3.10+
- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- matplotlib
- mlxtend (for association mining)

## 📁 Data Files

- `maharashtra_rainfall_clean_rebuilt.csv` — Main dataset (121,292 records)
- `monthly_rainfall_analysis.csv` — Aggregated monthly data
- `regression_comparison.csv` — Pre-computed model metrics

## 💡 For Demonstration

Navigate through pages in sequence to show:
1. How raw data is cleaned and prepared
2. Three types of regression with improving complexity
3. Market-basket analysis techniques on rainfall patterns
4. Classification trees with different splitting criteria

Each section ends with insights that show conceptual understanding, not just code execution.

---

**Dataset Source:** Maharashtra State Water Resources Department (2021-2025)  
**Project Type:** Data Analytics Course Project  
**Dashboard Framework:** Streamlit
