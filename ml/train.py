"""
Training Pipeline

This script:
1. Loads raw dataset
2. Performs K-fold cross validation
3. Selects SHAP-important features inside each fold
4. Finds stable features across folds
5. Optimizes classification threshold
6. Trains final production model
7. Saves all artifacts for API usage

Run manually before starting API:
    python -m ml.train
"""

import os
import json
import pandas as pd
import numpy as np
import shap
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    recall_score
)


def train_pipeline(
    data_path="TZ.csv",
    artifacts_dir="artifacts"
):
    os.makedirs(artifacts_dir, exist_ok=True)

    # ===============================
    # 1. Load dataset
    # ===============================
    df = pd.read_csv(data_path)

    TARGET = "ушел_из_банка"
    DROP_COLS = ["ID", "ID_клиента", "фамилия"]

    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET].astype(int)

    categorical_features = [
        "кредитный_рейтинг",
        "город",
        "пол",
        "активный_клиент",
        "есть_кредитка"
    ]

    for col in categorical_features:
        X[col] = X[col].astype(str)

    # ===============================
    # 2. Cross-validation setup
    # ===============================
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y))
    selected_features_per_fold = []
    
    oof_probs = np.zeros(len(y))
    oof_true = np.zeros(len(y))

    # ===============================
    # 3. CV with SHAP feature selection
    # ===============================
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        print(f"Fold {fold+1}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Base model (all features)
        base_model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            cat_features=categorical_features,
            verbose=0,
            random_state=42
        )
        base_model.fit(X_train, y_train)

        # SHAP importance
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer(X_train)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

        shap_df = (
            pd.DataFrame({
                "feature": X_train.columns,
                "importance": mean_abs_shap
            })
            .sort_values("importance", ascending=False)
        )

        shap_df["cumsum"] = shap_df["importance"].cumsum() / shap_df["importance"].sum()

        fold_features = shap_df.loc[
            shap_df["cumsum"] <= 0.95, "feature"
        ].tolist()

        if len(fold_features) < 5:
            fold_features = shap_df["feature"].head(5).tolist()

        selected_features_per_fold.append(fold_features)

        # Train reduced model
        X_train_sel = X_train[fold_features]
        X_val_sel = X_val[fold_features]

        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            class_weights=[1, pos_weight],
            cat_features=[f for f in fold_features if f in categorical_features],
            verbose=0,
            random_state=42
        )

        #model.fit(X_train_sel, y_train)
        #oof_probs[val_idx] = model.predict_proba(X_val_sel)[:, 1]
        model.fit(X_train_sel, y_train)
        oof_probs[val_idx] = model.predict_proba(X_val_sel)[:, 1]
        oof_true[val_idx] = y_val

    # ===============================
    # 4. Stable features
    # ===============================
    feature_counter = Counter(
        f for fold_feats in selected_features_per_fold for f in fold_feats
    )

    stable_features = [
        f for f, c in feature_counter.items()
        if c >= 3
    ]

    print("Stable features:", stable_features)
    
    # ===============================
    # 6. CV metrics (using stable features)
    # ===============================
    roc_auc = roc_auc_score(oof_true, oof_probs)
    pr_auc = average_precision_score(oof_true, oof_probs)

    print(f"\nOOF ROC-AUC : {roc_auc:.4f}")
    print(f"OOF PR-AUC  : {pr_auc:.4f}")

    # ===============================
    # 5. Threshold optimization
    # ===============================
    precision, recall, thresholds = precision_recall_curve(y, oof_probs)
    valid = precision[:-1] >= 0.30
    best_idx = np.argmax(recall[:-1][valid])
    best_threshold = thresholds[valid][best_idx]
    y_pred = (oof_probs >= best_threshold).astype(int)

    print("\nOptimal threshold")
    print(f"Threshold  : {best_threshold:.3f}")
    print(f"Recall     : {recall[:-1][valid][best_idx]:.3f}")
    print(f"Precision  : {precision[:-1][valid][best_idx]:.3f}")
    print(f"Final Recall: {recall_score(oof_true, y_pred):.3f}")

    # ===============================
    # 6. Final production model
    # ===============================
    X_final = X[stable_features]

    final_model = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        cat_features=[f for f in stable_features if f in categorical_features],
        verbose=0,
        random_state=42
    )

    final_model.fit(X_final, y)

    # Save artifacts
    final_model.save_model(f"{artifacts_dir}/model.cbm")

    json.dump(stable_features,
              open(f"{artifacts_dir}/stable_features.json", "w"))

    json.dump({"threshold": float(best_threshold)},
              open(f"{artifacts_dir}/threshold.json", "w"))

    json.dump(categorical_features,
              open(f"{artifacts_dir}/categorical_features.json", "w"))

    # Global feature importance
    importance = dict(zip(
        stable_features,
        final_model.get_feature_importance()
    ))

    json.dump(importance,
              open(f"{artifacts_dir}/feature_importance.json", "w"))

    print("Training complete. Artifacts saved.")


if __name__ == "__main__":
    train_pipeline()

