"""Time-aware split, model training and the soft-voting ensemble."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .costs import best_threshold
from .features import RateEncoder, build_features


def time_split(df: pd.DataFrame, train=0.6, val=0.2):
    """Chronological train / validation / test. The test set is always the latest claims."""
    df = df.sort_values("claim_date", kind="stable").reset_index(drop=True)
    a, b = int(len(df) * train), int(len(df) * (train + val))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def score_features(models: dict, X: pd.DataFrame) -> dict:
    """Per-model probabilities plus the soft-voting ensemble (plain mean)."""
    s = {k: m.predict_proba(X)[:, 1] for k, m in models.items()}
    s["ensemble"] = np.mean([s["xgboost"], s["random_forest"]], axis=0)
    return s


def fit_bundle(df: pd.DataFrame, n_estimators: int = 300, seed: int = 42) -> dict:
    """Fit encoder + models on the train split, choose the cost threshold on the validation split."""
    train, val, _ = time_split(df)
    enc = RateEncoder().fit(train)
    X_tr = build_features(train, enc.oof(train, seed=seed))
    y_tr = train["is_fraud"].to_numpy()

    # imbalance: softened class weight for XGBoost, balanced sub-sample weights for the forest
    spw = float(np.sqrt((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)))
    models = {
        "xgboost": XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            n_jobs=2,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=seed,
        ),
    }
    for m in models.values():
        m.fit(X_tr, y_tr)

    X_val = build_features(val, enc.transform(val))
    thr, _ = best_threshold(score_features(models, X_val)["ensemble"], val["is_fraud"], val["claim_amount"])
    return {"encoder": enc, "models": models, "columns": list(X_tr.columns), "threshold": thr}
