import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


def fit_km(
    duration: pd.Series, event: pd.Series, label: str = "All"
) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(duration, event_observed=event, label=label)
    return kmf


def km_trace(kmf: KaplanMeierFitter) -> dict:
    """Extract KM data as plain Python lists for Plotly."""
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_survival_function_
    return {
        "label": str(sf.columns[0]),
        "timeline": sf.index.tolist(),
        "survival": sf.iloc[:, 0].tolist(),
        "ci_lower": ci.iloc[:, 0].tolist(),
        "ci_upper": ci.iloc[:, 1].tolist(),
    }


def logrank_p(
    df: pd.DataFrame, group_col: str, group_a: str, group_b: str
) -> float:
    ga = df[df[group_col] == group_a]
    gb = df[df[group_col] == group_b]
    result = logrank_test(
        ga["tenure"],
        gb["tenure"],
        event_observed_A=ga["Churn"],
        event_observed_B=gb["Churn"],
    )
    return float(result.p_value)


def fit_cox(df_encoded: pd.DataFrame) -> CoxPHFitter:
    """Fit Cox PH on the fully encoded dataframe (tenure = duration, Churn = event)."""
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_encoded, duration_col="tenure", event_col="Churn", show_progress=False)
    return cph


def cox_summary(cph: CoxPHFitter) -> pd.DataFrame:
    """Return hazard ratios + 95 % CI + p-value, sorted by HR."""
    s = cph.summary[
        ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    ].copy()
    s.columns = ["HR", "CI_lower", "CI_upper", "p_value"]
    return s.sort_values("HR")


def predict_survival_fn(cph: CoxPHFitter, customer_enc: pd.DataFrame) -> pd.DataFrame:
    """
    Predict the survival function for a single encoded customer.
    lifelines selects only the fitted covariate columns from customer_enc automatically.
    """
    return cph.predict_survival_function(customer_enc)
