import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

def train_and_predict_with_cv(
    X, y, X_test, lgb_params, n_splits=5, seed=42, shuffle=True,
    early_stopping_rounds=100, num_boost_round=1000, loss='binary_logloss'
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
        print(f"Fold {fold+1}")
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y[trn_idx], y[val_idx]

        # バリデーション側でもcategory型に変更（万が一object型のまま残っていたら対応）
        for col in object_cols:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')

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
        print(f"Fold {fold+1} AUC: {val_auc:.4f}")
        print(f"Fold {fold+1} {loss}: {val_loss:.4f}")

    overall_auc = roc_auc_score(y, oof_preds)
    overall_logloss = log_loss(y, oof_preds)
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Overall {loss}: {overall_logloss:.4f}")

    return models, oof_preds, test_preds, overall_auc, overall_logloss
