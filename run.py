import os
import json
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler 

from utils.logger import get_logger
from utils.basic_imputer import basic_impute
# from models.lightgbm_model import train_and_predict_with_cv
from models.lightgbm_model import train_and_predict_with_cv as lgb_train_cv
from models.xgboost_model import train_and_predict_with_cv as xgb_train_cv
from models.random_forest_model import train_and_predict_with_cv as rf_train_cv
from models.mlp_model import train_and_predict_with_cv as mlp_train_cv
from models.logistic_model import train_and_predict_with_cv as lr_train_cv

from scripts import data_cleaning, generate_basic_features
from sklearn.metrics import roc_auc_score, log_loss

if __name__ == '__main__':
    # ロガー準備
    logger, log_file = get_logger()

    # コンフィグ読み込み
    with open('config/default.json', 'r') as f:
        config = json.load(f)
    
    # コンフィグから設定取得
    features = config["features"]
    lgbm_params = config["lgbm_params"]
    xgb_params = config["xgb_params"]
    # rf_params = config["rf_params"]
    mlp_params = config["mlp_params"]
    # lr_params = config["lr_params"]

    target_name = config["target_name"]
    ID_name = config["ID_name"]
    loss = config["loss"]

    # データ読み込み
    train_path = 'data/input/train.csv'
    test_path = 'data/input/test.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 特徴量エンジニアリングで作った特徴量をgenerate
    train = generate_basic_features.generate_basic_features(train)
    test = generate_basic_features.generate_basic_features(test)
    # generate_basic_features(train)
    # generate_basic_features(test)

    # 前処理
    data_cleaning.clean(train)
    data_cleaning.clean(test)
    logger.info("Data cleaned")

    # 欠損値補完
    # train = basic_impute(train)
    # test = basic_impute(test)

    # ログ出力
    logger.info(f"Using features: {features}")
    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    logger.info(f"LGBM params: {lgbm_params}")
    logger.info(f"XGB params: {xgb_params}")
    # logger.info(f"RF params: {rf_params}")
    logger.info(f"MLP params: {mlp_params}")
    # logger.info(f"LR params: {lr_params}")

    # 学習用データ作成
    # 作った特徴量についてここでconcatする
    X = train[features]
    y = train[target_name].values
    X_test = test[features]

    # # 標準化
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X_test = scaler.transform(X_test)

    # MLP
    mlp_models, mlp_oof, mlp_test, mlp_auc, mlp_logloss = mlp_train_cv(
        X=X, y=y, X_test=X_test, mlp_params=mlp_params, n_splits=5, seed=42, shuffle=True, logger=logger
    )
    # XGBoost
    xgb_models, xgb_oof, xgb_test, xgb_auc, xgb_logloss = xgb_train_cv(
        X=X, y=y, X_test=X_test, xgb_params=xgb_params, n_splits=5, seed=42, shuffle=True, logger=logger
    )
    # LightGBM
    lgb_models, lgb_oof, lgb_test, lgb_auc, lgb_logloss = lgb_train_cv(
        X=X, y=y, X_test=X_test, lgb_params=lgbm_params, n_splits=5, seed=42, shuffle=True,
        early_stopping_rounds=100, num_boost_round=1000, loss=loss, logger=logger
    )
    

    # # RandomForest
    # rf_models, rf_oof, rf_test, rf_auc, rf_logloss = rf_train_cv(
    #     X=X, y=y, X_test=X_test, rf_params=rf_params, n_splits=5, seed=42, shuffle=True, logger=logger
    # )


    # # Logistic Regression
    # lr_models, lr_oof, lr_test, lr_auc, lr_logloss = lr_train_cv(
    #     X=X, y=y, X_test=X_test, lr_params=lr_params, n_splits=5, seed=42, shuffle=True, logger=logger
    # )

    # アンサンブル: 簡易的には複数モデルの平均を取る
    # ensemble_oof = (lgb_oof + xgb_oof + rf_oof + mlp_oof + lr_oof) / 5
    # ensemble_test = (lgb_test + xgb_test + rf_test + mlp_test + lr_test) / 5
    ensemble_oof = (lgb_oof + xgb_oof + mlp_oof) / 3
    ensemble_test = (lgb_test + xgb_test + mlp_oof) / 3

    # アンサンブルスコア計算
    ensemble_auc = roc_auc_score(y, ensemble_oof)
    ensemble_logloss = log_loss(y, ensemble_oof)
    logger.info(f"Ensemble Overall AUC: {ensemble_auc:.4f}")
    logger.info(f"Ensemble Overall Logloss: {ensemble_logloss:.4f}")

    now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    submission_file = f"data/output/submission_ensemble_{now}.csv"
    submission = pd.DataFrame({
        ID_name: test[ID_name],
        target_name: ensemble_test
    })
    submission.to_csv(submission_file, index=False)
    logger.info(f"Ensemble Submission saved to {submission_file}")
    logger.info(f"Logs saved to {log_file}")

