import os
import json
import pandas as pd
from datetime import datetime

from utils.logger import get_logger
from models.lightgbm_model import train_and_predict_with_cv

if __name__ == '__main__':
    # ロガー準備
    logger, log_file = get_logger()

    # コンフィグ読み込み
    with open('config/default.json', 'r') as f:
        config = json.load(f)
    
    # コンフィグから設定取得
    features = config["features"]
    lgbm_params = config["lgbm_params"]
    target_name = config["target_name"]
    ID_name = config["ID_name"]
    loss = config["loss"]

    # データ読み込み
    train_path = 'data/input/train.csv'
    test_path = 'data/input/test.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # ログ出力
    logger.info(f"Using features: {features}")
    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    logger.info(f"LGBM params: {lgbm_params}")

    # 学習用データ作成
    X = train[features]
    y = train[target_name].values
    X_test = test[features]

    # CV学習・予測
    models, oof_preds, test_preds, overall_auc, overall_logloss = train_and_predict_with_cv(
        X=X,
        y=y,
        X_test=X_test,
        lgb_params=lgbm_params,
        n_splits=5,
        seed=42,
        shuffle=True,
        early_stopping_rounds=100,
        num_boost_round=1000,
        loss=loss
    )

    # ログ出力(最終スコア)
    logger.info(f"Overall AUC: {overall_auc:.4f}")
    logger.info(f"Overall {loss}: {overall_logloss:.4f}")

    # 提出ファイル作成
    now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    submission_file = f"data/output/submission_{now}.csv"
    # バイナリ分類: predictはクラス1の確率を返すため、そのままTARGET列に利用
    submission = pd.DataFrame({
        ID_name: test[ID_name],
        target_name: test_preds
    })
    submission.to_csv(submission_file, index=False)
    logger.info(f"Submission saved to {submission_file}")
    logger.info(f"Logs saved to {log_file}")
