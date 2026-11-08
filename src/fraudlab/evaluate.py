"""Held-out evaluation: PR-AUC, recall at fixed precision, confusion matrix, cost,
false-positive analysis, permutation importance and the Monte Carlo sweep."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from .costs import net_savings
from .features import build_features
from .model import score_features, time_split
from .montecarlo import simulate


def recall_at_precision(y, scores, precision=0.5) -> float:
    p, r, _ = precision_recall_curve(y, scores)
    ok = p >= precision
    return float(r[ok].max()) if ok.any() else 0.0


def permutation_importance(models, X, y, n_repeats=3, seed=0) -> dict:
    """Drop in ensemble PR-AUC when one feature is shuffled."""
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, score_features(models, X)["ensemble"])
    out = {}
    for c in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(base - average_precision_score(y, score_features(models, Xp)["ensemble"]))
        out[c] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def false_positive_analysis(test: pd.DataFrame, flag: np.ndarray) -> dict:
    fp = test[flag & (test["is_fraud"] == 0)]
    tp = test[flag & (test["is_fraud"] == 1)]
    legit = (test["is_fraud"] == 0).sum()
    return {
        "false_positives": int(len(fp)),
        "false_positive_rate": float(len(fp) / legit),
        "share_of_flags_that_are_false": float(len(fp) / max(flag.sum(), 1)),
        "mean_amount_fp": float(fp["claim_amount"].mean()) if len(fp) else 0.0,
        "mean_amount_tp": float(tp["claim_amount"].mean()) if len(tp) else 0.0,
        "fp_by_injury_type": fp["injury_type"].value_counts(normalize=True).round(3).to_dict(),
        "fp_share_in_ring_garages": float(fp["garage_id"].str[1:].astype(int).lt(10).mean())
        if len(fp)
        else 0.0,
    }


def evaluate(bundle: dict, df: pd.DataFrame, mc_sims: int = 300, mc_claims: int = 50_000) -> dict:
    _, _, test = time_split(df)
    X = build_features(test, bundle["encoder"].transform(test))[bundle["columns"]]
    y, amount = test["is_fraud"].to_numpy(), test["claim_amount"].to_numpy()
    scores = score_features(bundle["models"], X)
    thr = bundle["threshold"]

    models = {
        k: {
            "pr_auc": float(average_precision_score(y, s)),
            "recall_at_precision_50": recall_at_precision(y, s, 0.5),
            "recall_at_precision_80": recall_at_precision(y, s, 0.8),
        }
        for k, s in scores.items()
    }
    flag = scores["ensemble"] >= thr
    tn, fp, fn, tp = confusion_matrix(y, flag).ravel()
    grid = sorted({*np.round(np.linspace(0.05, 0.95, 19), 2), round(thr, 4)})
    mc = simulate(scores["ensemble"], y, amount, grid, n_claims=mc_claims, n_sims=mc_sims)
    best = mc.loc[mc["savings_mean"].idxmax()]
    return {
        "n_test": int(len(y)),
        "test_fraud_rate": float(y.mean()),
        "models": models,
        "threshold": float(thr),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision_at_threshold": float(tp / max(tp + fp, 1)),
        "recall_at_threshold": float(tp / max(tp + fn, 1)),
        "net_savings_at_threshold": net_savings(y, amount, flag),
        "fraud_loss_without_model": float(amount[y == 1].sum()),
        "false_positive_analysis": false_positive_analysis(test, flag),
        "permutation_importance": permutation_importance(bundle["models"], X, y),
        "monte_carlo": {
            "best_threshold": float(best["threshold"]),
            "best_savings_mean": float(best["savings_mean"]),
            "best_savings_p5": float(best["savings_p5"]),
            "best_savings_p95": float(best["savings_p95"]),
            "baseline_annual_cost": float(best["baseline_cost_mean"]),
            "table": mc.round(1).assign(threshold=mc["threshold"].round(4)).to_dict(orient="records"),
        },
    }
