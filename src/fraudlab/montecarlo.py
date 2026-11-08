"""Monte Carlo of one year of claims: expected annual claim cost and fraud savings per threshold.

Each simulation bootstraps `n_claims` from the held-out set and draws uncertain inputs:
investigation cost (lognormal), claim-severity inflation (lognormal) and the share of
investigated frauds that are actually denied (Beta). The model scores are fixed."""

import numpy as np
import pandas as pd

from .costs import INVESTIGATION_COST


def simulate(scores, y, amount, thresholds, n_claims=50_000, n_sims=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scores, y, amount = np.asarray(scores), np.asarray(y), np.asarray(amount, dtype=float)
    thresholds = np.asarray(thresholds)
    cost = np.empty((n_sims, len(thresholds)))
    base = np.empty(n_sims)
    for s in range(n_sims):
        i = rng.integers(0, len(scores), n_claims)
        sc, fr, am = scores[i], y[i], amount[i] * rng.lognormal(0, 0.1)
        inv = INVESTIGATION_COST * rng.lognormal(0, 0.25)
        success = rng.beta(18, 2)
        base[s] = am.sum()
        for j, t in enumerate(thresholds):
            flag = sc >= t
            cost[s, j] = base[s] - success * am[flag & (fr == 1)].sum() + inv * flag.sum()
    sav = base[:, None] - cost
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "annual_cost_mean": cost.mean(0),
            "annual_cost_p5": np.quantile(cost, 0.05, axis=0),
            "annual_cost_p95": np.quantile(cost, 0.95, axis=0),
            "savings_mean": sav.mean(0),
            "savings_p5": np.quantile(sav, 0.05, axis=0),
            "savings_p95": np.quantile(sav, 0.95, axis=0),
            "baseline_cost_mean": base.mean(),
        }
    )
