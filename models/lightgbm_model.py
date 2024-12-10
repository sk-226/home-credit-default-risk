import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
import matplotlib.pyplot as plt
from datetime import datetime 
import os

def train_and_predict_with_cv(
    X, y, X_test, lgb_params, n_splits=5, seed=42, shuffle=True,
    early_stopping_rounds=100, num_boost_round=1000, loss='binary_logloss', logger=None
):
    """
    LightGBMによるクロスバリデーション実行関数（バイナリ分類用）。
    TODO: 前処理系の関数を別に作る
    """

    # object型カラムをcategory型へ変更
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
        logger.info(f"Fold {fold+1}")
        # copy()を使って明示的に独立したDataFrameを作成
        X_train = X.iloc[trn_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train, y_val = y[trn_idx], y[val_idx]

        train_dataset = lgb.Dataset(X_train, label=y_train)
        valid_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

        callbacks = []
        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, first_metric_only=False))
        callbacks.append(lgb.log_evaluation(100))

        model = lgb.train(
            params=lgb_params,
            train_set=train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[train_dataset, valid_dataset],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        models.append(model)

        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds

        test_fold_preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds += test_fold_preds / folds.n_splits

        val_auc = roc_auc_score(y_val, val_preds)
        val_loss = log_loss(y_val, val_preds)
        logger.info(f"Fold {fold+1} AUC: {val_auc:.4f}")
        logger.info(f"Fold {fold+1} {loss}: {val_loss:.4f}")

    overall_auc = roc_auc_score(y, oof_preds)
    overall_logloss = log_loss(y, oof_preds)
    logger.info(f"Overall AUC: {overall_auc:.4f}")
    logger.info(f"Overall {loss}: {overall_logloss:.4f}")

    now = datetime.now().strftime('%Y-%m-%d-%H-%M')
    roc_curve_file = f'results_fig/roc_curve_{now}.png'
    feature_importances_file = f'results_fig/feature_importances_{now}.png'
    os.makedirs('results_fig', exist_ok=True)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, val_preds)

    # Plot ROC curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Validation ROC curve (area = %0.4f)' % overall_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_curve_file)

    # Feature importance
    importances = model.feature_importance()
    feature_names = X_train.columns
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importances.sort_values('importance', ascending=False, inplace=True)
    num_features = len(feature_names)

    # Plot feature importance
    plt.figure(figsize=(10,12))
    plt.barh(feature_importances['feature'].iloc[:num_features][::-1], feature_importances['importance'].iloc[:50][::-1], color='steelblue')
    plt.xlabel('Importance')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig(feature_importances_file)

    return models, oof_preds, test_preds, overall_auc, overall_logloss
