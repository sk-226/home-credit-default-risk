import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.neural_network import MLPClassifier

from utils.basic_imputer import basic_impute

def train_and_predict_with_cv(
    X, y, X_test, mlp_params, n_splits=5, seed=42, shuffle=True, logger=None
):
    # 欠損値補完
    X = basic_impute(X)
    X_test = basic_impute(X_test)
    
    # カテゴリ列の処理
    # 数値列とカテゴリ列を分ける（例：object型をカテゴリとみなす）
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # trainとtestを結合してget_dummiesすると、trainとtestが同一のダミー列を持ちやすい
    # ID列などがあれば除外する必要あり
    X_test['is_test'] = 1
    X['is_test'] = 0
    combined = pd.concat([X, X_test], axis=0)
    
    # One-Hotエンコーディング
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

    # 再分割
    X = combined[combined['is_test'] == 0].drop('is_test', axis=1)
    X_test = combined[combined['is_test'] == 1].drop('is_test', axis=1)

    folds = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        if logger:
            logger.info(f"[MLP] Fold {fold+1}")
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y[trn_idx], y[val_idx]

        model = MLPClassifier(**mlp_params, random_state=seed)
        model.fit(X_train, y_train)
        
        models.append(model)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        
        val_auc = roc_auc_score(y_val, val_preds)
        val_loss = log_loss(y_val, val_preds)
        if logger:
            logger.info(f"[MLP] Fold {fold+1} AUC: {val_auc:.4f}")
            logger.info(f"[MLP] Fold {fold+1} Logloss: {val_loss:.4f}")

    overall_auc = roc_auc_score(y, oof_preds)
    overall_logloss = log_loss(y, oof_preds)
    if logger:
        logger.info(f"[MLP] Overall AUC: {overall_auc:.4f}")
        logger.info(f"[MLP] Overall Logloss: {overall_logloss:.4f}")

    return models, oof_preds, test_preds, overall_auc, overall_logloss
