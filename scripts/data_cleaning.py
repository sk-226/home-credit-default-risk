import numpy as np
import pandas as pd

# DAYS_EMPLOYEDのハズレ値をnanに置き換える
# FLAG_OWN_CAR, FLAG_OWN_REALTYを数値に変換する
def clean(df: pd.DataFrame) -> None:
    # DAYS_EMPLOYED anomalies
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)

    # Encode binary categorical features
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
            df[col].replace({"Y": 1, "N": 0}, inplace=True)
            df[col] = df[col].astype(float)
#     return df
