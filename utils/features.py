import pandas as pd
import numpy as np

FEATURE_COLS = [
    "amount", "amount_log", "day_of_week", "month", "is_weekend",
    "dept_avg_amount", "dept_std_amount", "emp_txn_count", "emp_avg_amount",
    "vendor_txn_count", "vendor_avg_amount", "amount_z_score",
    "cat_avg_amount", "amount_vs_dept_avg", "amount_vs_cat_avg",
    "dept_enc", "cat_enc", "vendor_enc",
    "is_round_number", "high_value_flag",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Basic temporal
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # Amount transforms
    df["amount_log"]   = np.log1p(df["amount"])
    df["is_round_number"] = (df["amount"] % 100 == 0).astype(int)
    df["high_value_flag"] = (df["amount"] > df["amount"].quantile(0.90)).astype(int)

    # Department stats
    dept_stats = df.groupby("department")["amount"].agg(["mean","std"]).rename(
        columns={"mean":"dept_avg_amount","std":"dept_std_amount"})
    df = df.join(dept_stats, on="department")

    # Employee stats
    emp_stats = df.groupby("employee_id")["amount"].agg(
        emp_txn_count="count", emp_avg_amount="mean")
    df = df.join(emp_stats, on="employee_id")

    # Vendor stats
    vend_stats = df.groupby("vendor")["amount"].agg(
        vendor_txn_count="count", vendor_avg_amount="mean")
    df = df.join(vend_stats, on="vendor")

    # Category stats
    cat_stats = df.groupby("category")["amount"].agg(cat_avg_amount="mean")
    df = df.join(cat_stats, on="category")

    # Relative features
    df["amount_z_score"]     = (df["amount"] - df["dept_avg_amount"]) / (df["dept_std_amount"] + 1e-6)
    df["amount_vs_dept_avg"] = df["amount"] / (df["dept_avg_amount"] + 1e-6)
    df["amount_vs_cat_avg"]  = df["amount"] / (df["cat_avg_amount"] + 1e-6)

    # Label encodings
    df["dept_enc"]   = df["department"].astype("category").cat.codes
    df["cat_enc"]    = df["category"].astype("category").cat.codes
    df["vendor_enc"] = df["vendor"].astype("category").cat.codes

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    return df
