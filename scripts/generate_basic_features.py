import os
import numpy as np
import pandas as pd


# 新しく作成した特徴量のリスト
# "CREDIT_INCOME_RATIO",
# "ANNUITY_INCOME_RATIO",
# "INCOME_PER_CHILD",
# "INCOME_PER_FAM_MEMBER",
# "CREDIT_TERM",
# "DAYS_EMPLOYED_PERC",
# "DAYS_REGISTRATION_PERC",
# "DAYS_ID_PUBLISH_PERC",
# "AGE_IN_YEARS",
# "YEARS_EMPLOYED"

# def generate_basic_features(df: pd.DataFrame) -> None:
def generate_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (1 + df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (1 + df["AMT_INCOME_TOTAL"])
    df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (1 + df["CNT_CHILDREN"])
    df["INCOME_PER_FAM_MEMBER"] = df["AMT_INCOME_TOTAL"] / (1 + df["CNT_FAM_MEMBERS"])
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (1 + df["AMT_CREDIT"])
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / (1 + df["DAYS_BIRTH"])
    df["DAYS_REGISTRATION_PERC"] = df["DAYS_REGISTRATION"] / (1 + df["DAYS_BIRTH"])
    df["DAYS_ID_PUBLISH_PERC"] = df["DAYS_ID_PUBLISH"] / (1 + df["DAYS_BIRTH"])
    df["AGE_IN_YEARS"] = (-df["DAYS_BIRTH"] / 365).astype(float)
    df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365).astype(float)
    return df


def save_features(df: pd.DataFrame, filename: str) -> None:
    os.makedirs("features", exist_ok=True)
    df.to_csv(os.path.join("features", filename), index=False)
