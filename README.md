# Customer Churn Intelligence Dashboard

An end-to-end churn prediction project combining **classical ML** (Gradient Boosted Trees + SHAP) with **survival analysis** (Cox Proportional Hazards), served through an interactive Streamlit dashboard.

> Most churn models answer *"will this customer leave?"*  
> This project also answers **"when?"** — using survival analysis to model time-to-churn with right-censored observations.

---

## Methodology

### 1 · Binary Classifier — Gradient Boosted Trees
- `HistGradientBoostingClassifier` (sklearn) trained on all 19 features  
- Evaluated with AUC-ROC, F1, Precision, Recall  
- **SHAP** (TreeExplainer) for global feature importance (beeswarm) and individual explanations (waterfall)

### 2 · Survival Analysis — Cox Proportional Hazards
- **Kaplan-Meier** curves for overall retention and by customer segment (contract type, internet service)  
- **Log-rank test** to confirm statistical significance between segments  
- **Cox PH model** (lifelines, ridge penalizer) estimating hazard ratios for all 29 covariates  
- Individual survival curve prediction: expected tenure until churn for any given customer profile

---

## Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.839** |
| F1 Score | 0.576 |
| Accuracy | 0.794 |
| Log-rank p (Month-to-month vs Two year) | < 0.001 |

Key findings from the Cox PH model:
- **Two-year contract**: HR = 0.35 — the strongest protective factor
- **Fiber optic internet**: HR = 1.60 — highest churn risk driver
- **Electronic check payment**: HR = 1.47 — significant risk signal

---

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM Watson Analytics dataset  
7,043 customers · 19 features · 26.5% churn rate

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Data Explorer** | KPIs, tenure distribution, churn rate by contract, charges vs tenure scatter |
| **ML Classifier** | ROC curve, confusion matrix, SHAP global importance, SHAP beeswarm |
| **Survival Analysis** | KM curves overall + by segment, median survival times, Cox PH forest plot |
| **Predict** | Customer form → churn probability gauge + individual survival curve + SHAP waterfall |

---

## Project Structure

```
churn-prediction/
├── app.py               # Streamlit dashboard (4 tabs)
├── download_data.py     # Fetch dataset from IBM's public repository
├── requirements.txt
└── src/
    ├── data.py          # Load, clean, encode, train/test split
    ├── classifier.py    # GBM training, evaluation, SHAP computation
    └── survival.py      # Kaplan-Meier, log-rank test, Cox PH, predictions
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset
python download_data.py

# 3. Launch the dashboard
streamlit run app.py
```

Open http://localhost:8501 — models train and cache on first load (~30 s).

---

## Tech Stack

`Python` · `scikit-learn` · `lifelines` · `SHAP` · `Streamlit` · `Plotly`
