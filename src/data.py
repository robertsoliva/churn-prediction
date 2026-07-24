import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes, encode target, drop ID. Keeps categorical columns as-is for survival analysis."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns=["customerID"])


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all object columns (drop_first to avoid multicollinearity)."""
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def get_xy(df_encoded: pd.DataFrame):
    """Split encoded dataframe into train/test feature matrices and targets."""
    y = df_encoded["Churn"]
    X = df_encoded.drop(columns=["Churn"])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
