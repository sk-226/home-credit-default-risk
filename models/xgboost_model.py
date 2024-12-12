import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier


def train_and_predict_with_cv(
    X, y, X_test, xgb_params, n_splits=5, seed=42, shuffle=True, logger=None
):
    # object型をcategory型に変換
    object_cols = X.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        X[col] = X[col].astype('category')
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')

    folds = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        logger.info(f"[XGB] Fold {fold+1}")
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y[trn_idx], y[val_idx]

        # xgb_paramsに `enable_categorical`: True, `tree_method`: 'hist' を追加している前提
        model = XGBClassifier(**xgb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )
        
        models.append(model)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        
        val_auc = roc_auc_score(y_val, val_preds)
        val_loss = log_loss(y_val, val_preds)
        logger.info(f"[XGB] Fold {fold+1} AUC: {val_auc:.4f}")
        logger.info(f"[XGB] Fold {fold+1} Logloss: {val_loss:.4f}")

    overall_auc = roc_auc_score(y, oof_preds)
    overall_logloss = log_loss(y, oof_preds)
    logger.info(f"[XGB] Overall AUC: {overall_auc:.4f}")
    logger.info(f"[XGB] Overall Logloss: {overall_logloss:.4f}")

    return models, oof_preds, test_preds, overall_auc, overall_logloss
