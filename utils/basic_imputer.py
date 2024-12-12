import pandas as pd
from sklearn.impute import SimpleImputer

def basic_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレームに含まれる数値およびカテゴリカル変数に対して欠損値補完を行う関数。
    数値列は中央値で補完し、カテゴリ列は最頻値で補完します。
    
    Parameters
    ----------
    df : pd.DataFrame
        補完対象のデータフレーム
    
    Returns
    -------
    pd.DataFrame
        補完後のデータフレーム
    """
    # 元のDataFrameをコピー
    df = df.copy()

    # 数値列とカテゴリ列を分離
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # 数値列の欠損補完 (中央値)
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # カテゴリ列の欠損補完 (最頻値)
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df
