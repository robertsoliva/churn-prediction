import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from pathlib import Path

from src.classifier import compute_shap, evaluate, train as train_clf
from src.data import clean, encode, get_xy, load_raw
from src.survival import (
    cox_summary,
    fit_cox,
    fit_km,
    km_trace,
    logrank_p,
    predict_survival_fn,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PALETTE = {
    "churned": "#E63946",
    "retained": "#2A9D8F",
    "accent": "#457B9D",
    "warn": "#F4A261",
}

# ── Load & cache everything ────────────────────────────────────────────────────
DATA_PATH = Path("data/telco_churn.csv")


@st.cache_resource(show_spinner="Training models — ~30 s on first run…")
def load_everything():
    df_raw = load_raw(str(DATA_PATH))
    df_clean = clean(df_raw)
    df_enc = encode(df_clean)
    X_train, X_test, y_train, y_test = get_xy(df_enc)
    clf = train_clf(X_train, y_train)
    cox = fit_cox(df_enc)
    metrics = evaluate(clf, X_test, y_test)
    explainer, shap_vals, X_shap = compute_shap(clf, X_test)
    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "df_enc": df_enc,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "clf": clf,
        "cox": cox,
        "metrics": metrics,
        "explainer": explainer,
        "shap_vals": shap_vals,
        "X_shap": X_shap,
        "feature_cols": X_train.columns.tolist(),
    }


if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Run `python download_data.py` in your terminal first, "
        "then refresh this page."
    )
    st.stop()

d = load_everything()
df_raw: pd.DataFrame = d["df_raw"]
df_clean: pd.DataFrame = d["df_clean"]
df_enc: pd.DataFrame = d["df_enc"]
X_train, X_test = d["X_train"], d["X_test"]
y_train, y_test = d["y_train"], d["y_test"]
clf = d["clf"]
cox = d["cox"]
metrics: dict = d["metrics"]
explainer = d["explainer"]
shap_vals = d["shap_vals"]
X_shap: pd.DataFrame = d["X_shap"]
feature_cols: list[str] = d["feature_cols"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Intelligence Dashboard")
st.markdown(
    "**Dataset:** Telco Customer Churn (IBM) — 7,043 customers · 19 features&emsp;|&emsp;"
    "**Models:** Gradient Boosted Trees (sklearn) + SHAP &nbsp;·&nbsp; Cox Proportional Hazards"
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Data Explorer", "🤖 ML Classifier", "📈 Survival Analysis", "🎯 Predict"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    n_total = len(df_clean)
    n_churned = int(df_clean["Churn"].sum())
    churn_rate = n_churned / n_total

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{n_total:,}")
    c2.metric("Churned", f"{n_churned:,}")
    c3.metric("Churn Rate", f"{churn_rate:.1%}")
    c4.metric("Avg Tenure", f"{df_clean['tenure'].mean():.1f} mo")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            values=[n_total - n_churned, n_churned],
            names=["Retained", "Churned"],
            color_discrete_sequence=[PALETTE["retained"], PALETTE["churned"]],
            title="Churn Distribution",
            hole=0.35,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        label_map = {0: "Retained", 1: "Churned"}
        fig = px.histogram(
            df_clean,
            x="tenure",
            color=df_clean["Churn"].map(label_map),
            nbins=36,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={"Retained": PALETTE["retained"], "Churned": PALETTE["churned"]},
            title="Tenure Distribution by Churn Status",
            labels={"tenure": "Tenure (months)", "color": "Status"},
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        contract_churn = (
            df_raw.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean())
            .reset_index()
        )
        contract_churn.columns = ["Contract", "Churn Rate"]
        contract_churn["Label"] = contract_churn["Churn Rate"].map("{:.1%}".format)
        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale="RdYlGn_r",
            text="Label",
            title="Churn Rate by Contract Type",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            df_clean,
            x="tenure",
            y="MonthlyCharges",
            color=df_clean["Churn"].map(label_map),
            color_discrete_map={"Retained": PALETTE["retained"], "Churned": PALETTE["churned"]},
            opacity=0.35,
            title="Monthly Charges vs Tenure",
            labels={"color": "Status", "tenure": "Tenure (months)"},
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC-ROC", f"{metrics['auc']:.3f}")
    c2.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c3.metric("Precision", f"{metrics['precision']:.3f}")
    c4.metric("Recall", f"{metrics['recall']:.3f}")
    c5.metric("F1 Score", f"{metrics['f1']:.3f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=metrics["fpr"],
                y=metrics["tpr"],
                mode="lines",
                name=f"GBM  AUC = {metrics['auc']:.3f}",
                line=dict(color=PALETTE["accent"], width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Random",
            )
        )
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cm = metrics["cm"]
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Predicted: Retained", "Predicted: Churned"],
            y=["Actual: Retained", "Actual: Churned"],
            title="Confusion Matrix",
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("SHAP Feature Importance")
    col1, col2 = st.columns(2)

    with col1:
        mean_abs = np.abs(shap_vals.values).mean(axis=0)
        df_imp = (
            pd.DataFrame({"Feature": X_shap.columns, "Mean |SHAP|": mean_abs})
            .sort_values("Mean |SHAP|", ascending=True)
            .tail(15)
        )
        fig = px.bar(
            df_imp,
            x="Mean |SHAP|",
            y="Feature",
            orientation="h",
            color="Mean |SHAP|",
            color_continuous_scale="Reds",
            title="Top 15 Features — Global SHAP Importance",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**SHAP Beeswarm** — impact distribution per feature")
        plt.figure()
        shap.plots.beeswarm(shap_vals, max_display=15, show=False)
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close("all")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        """
        **Cox Proportional Hazards** reframes churn as a *time-to-event* problem.
        Rather than predicting *whether* a customer churns, it models *when* —
        treating customers who haven't churned yet as **right-censored** observations
        (they contribute information up to their last observed tenure).

        The **Hazard Ratio (HR)** quantifies the multiplicative effect on instantaneous
        churn risk: HR = 2 → twice the risk; HR = 0.5 → half the risk, all else equal.
        """
    )
    st.divider()

    # ── Kaplan-Meier curves ────────────────────────────────────────────────────
    st.subheader("Kaplan-Meier Survival Curves")
    col1, col2 = st.columns(2)

    with col1:
        kmf_all = fit_km(df_clean["tenure"], df_clean["Churn"], label="All Customers")
        t = km_trace(kmf_all)
        fig = go.Figure()
        # Confidence band (fill between upper and lower)
        fig.add_trace(
            go.Scatter(
                x=t["timeline"] + t["timeline"][::-1],
                y=t["ci_upper"] + t["ci_lower"][::-1],
                fill="toself",
                fillcolor=f"rgba(69,123,157,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t["timeline"],
                y=t["survival"],
                mode="lines",
                name=t["label"],
                line=dict(color=PALETTE["accent"], width=2),
            )
        )
        fig.update_layout(
            title="Overall Survival (Retention) Curve",
            xaxis_title="Tenure (months)",
            yaxis_title="P(still subscribed)",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        colors_contract = {
            "Month-to-month": PALETTE["churned"],
            "One year": PALETTE["warn"],
            "Two year": PALETTE["retained"],
        }
        fig = go.Figure()
        for ctype in df_clean["Contract"].unique():
            sub = df_clean[df_clean["Contract"] == ctype]
            kmf = fit_km(sub["tenure"], sub["Churn"], label=ctype)
            t = km_trace(kmf)
            color = colors_contract.get(ctype, PALETTE["accent"])
            fig.add_trace(
                go.Scatter(
                    x=t["timeline"],
                    y=t["survival"],
                    mode="lines",
                    name=t["label"],
                    line=dict(color=color, width=2),
                )
            )
        p_val = logrank_p(df_clean, "Contract", "Month-to-month", "Two year")
        p_str = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
        fig.update_layout(
            title=f"Survival by Contract Type  (log-rank {p_str})",
            xaxis_title="Tenure (months)",
            yaxis_title="P(still subscribed)",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── KM by Internet Service ─────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        colors_inet = {
            "Fiber optic": PALETTE["churned"],
            "DSL": PALETTE["warn"],
            "No": PALETTE["retained"],
        }
        fig = go.Figure()
        for itype in df_clean["InternetService"].unique():
            sub = df_clean[df_clean["InternetService"] == itype]
            kmf = fit_km(sub["tenure"], sub["Churn"], label=itype)
            t = km_trace(kmf)
            fig.add_trace(
                go.Scatter(
                    x=t["timeline"],
                    y=t["survival"],
                    mode="lines",
                    name=t["label"],
                    line=dict(color=colors_inet.get(itype, PALETTE["accent"]), width=2),
                )
            )
        p_inet = logrank_p(df_clean, "InternetService", "Fiber optic", "No")
        p_str_inet = f"p < 0.001" if p_inet < 0.001 else f"p = {p_inet:.4f}"
        fig.update_layout(
            title=f"Survival by Internet Service  (log-rank {p_str_inet})",
            xaxis_title="Tenure (months)",
            yaxis_title="P(still subscribed)",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Median survival by contract segment
        rows = []
        for ctype in ["Month-to-month", "One year", "Two year"]:
            sub = df_clean[df_clean["Contract"] == ctype]
            kmf = fit_km(sub["tenure"], sub["Churn"], label=ctype)
            med = kmf.median_survival_time_
            rows.append({"Contract": ctype, "Median Tenure Until Churn (mo)": med})
        df_med = pd.DataFrame(rows)
        fig = px.bar(
            df_med,
            x="Contract",
            y="Median Tenure Until Churn (mo)",
            color="Contract",
            color_discrete_map={
                "Month-to-month": PALETTE["churned"],
                "One year": PALETTE["warn"],
                "Two year": PALETTE["retained"],
            },
            title="Median Survival Time by Contract Type",
            text="Median Tenure Until Churn (mo)",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Cox PH Forest Plot ─────────────────────────────────────────────────────
    st.subheader("Cox PH — Hazard Ratio Forest Plot")
    st.caption(
        "Showing the 25 most significant covariates (sorted by p-value). "
        "Confidence intervals are 95%. Reference categories are implicit (all-zero encoding)."
    )

    hr_df = cox_summary(cox)
    top_hr = hr_df.sort_values("p_value").head(25).sort_values("HR")

    dot_colors = [
        PALETTE["churned"] if hr > 1 else PALETTE["retained"]
        for hr in top_hr["HR"]
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=top_hr["HR"],
            y=top_hr.index,
            mode="markers",
            marker=dict(size=10, color=dot_colors),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(top_hr["CI_upper"] - top_hr["HR"]).tolist(),
                arrayminus=(top_hr["HR"] - top_hr["CI_lower"]).tolist(),
                color="gray",
            ),
            name="HR",
        )
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="HR = 1")
    fig.update_layout(
        title="Hazard Ratio Forest Plot — Cox Proportional Hazards",
        xaxis_title="Hazard Ratio (HR)",
        height=600,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Score an Individual Customer")
    st.markdown(
        "Fill in the customer profile and hit **Predict** to get the churn probability "
        "(XGBoost), their predicted survival curve (Cox PH), and a SHAP explanation."
    )

    form_col, result_col = st.columns([1, 2])

    with form_col:
        with st.form("predict_form"):
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total = st.number_input(
                "Total Charges ($)", 0.0, 9000.0, float(tenure * monthly)
            )
            submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        with result_col:
            # Build raw customer dict and encode it
            customer_raw = pd.DataFrame(
                [
                    {
                        "gender": gender,
                        "SeniorCitizen": 1 if senior == "Yes" else 0,
                        "Partner": partner,
                        "Dependents": dependents,
                        "tenure": tenure,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless,
                        "PaymentMethod": payment,
                        "MonthlyCharges": monthly,
                        "TotalCharges": total,
                    }
                ]
            )
            cat_cols = customer_raw.select_dtypes(include="object").columns.tolist()
            customer_enc = pd.get_dummies(customer_raw, columns=cat_cols)
            # Reindex to match training columns — missing dummies (= reference categories)
            # are correctly set to 0
            customer_enc = customer_enc.reindex(columns=feature_cols, fill_value=0)

            # ── XGBoost prediction ──────────────────────────────────────────
            churn_prob = float(clf.predict_proba(customer_enc)[0][1])
            risk_label = "High Risk" if churn_prob > 0.5 else "Low Risk"
            gauge_color = PALETTE["churned"] if churn_prob > 0.5 else PALETTE["retained"]

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    number={"suffix": "%", "font": {"size": 52}},
                    title={"text": f"Churn Probability — {risk_label}"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, 60], "color": "#fff3cd"},
                            {"range": [60, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": churn_prob * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Cox PH survival curve ───────────────────────────────────────
            sf = predict_survival_fn(cox, customer_enc)
            sf_series = sf.iloc[:, 0]

            # Median survival time (where curve crosses 0.5)
            below_half = sf_series[sf_series <= 0.5]
            if not below_half.empty:
                median_t = int(below_half.index[0])
                median_note = f"Median predicted tenure until churn: **{median_t} months**"
            else:
                median_note = "Survival probability stays above 50 % throughout the observation window."

            fig_sf = go.Figure()
            fig_sf.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="50%")
            fig_sf.add_trace(
                go.Scatter(
                    x=sf_series.index.tolist(),
                    y=sf_series.values.tolist(),
                    mode="lines",
                    line=dict(color=gauge_color, width=2.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({','.join(str(int(gauge_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.1)",
                    name="Survival probability",
                )
            )
            fig_sf.update_layout(
                title="Predicted Survival Curve (Cox PH)",
                xaxis_title="Tenure (months)",
                yaxis_title="P(still subscribed)",
                yaxis_range=[0, 1],
            )
            st.plotly_chart(fig_sf, use_container_width=True)
            st.markdown(median_note)

            # ── SHAP waterfall ──────────────────────────────────────────────
            st.markdown("**SHAP Explanation — why this score?**")
            customer_shap = explainer(customer_enc)
            plt.figure()
            shap.plots.waterfall(customer_shap[0], max_display=12, show=False)
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close("all")
    else:
        with result_col:
            st.info("Fill in the customer profile on the left and click **Predict**.")
