"""Cost model.

Flagging a claim costs one investigation; investigating a fraudulent claim avoids paying it
(with probability `success`). Net savings = avoided payouts - investigation spend."""

import numpy as np

INVESTIGATION_COST = 200.0


def net_savings(y, amount, flag, inv_cost=INVESTIGATION_COST, success=1.0) -> float:
    y, amount, flag = np.asarray(y), np.asarray(amount), np.asarray(flag, dtype=bool)
    return float(success * amount[flag & (y == 1)].sum() - inv_cost * flag.sum())


def best_threshold(scores, y, amount, inv_cost=INVESTIGATION_COST, grid=None):
    """Score threshold that maximises net savings. Returns (threshold, savings)."""
    scores = np.asarray(scores)
    if grid is None:
        grid = np.unique(np.quantile(scores, np.linspace(0.5, 0.9995, 300)))
    sav = [net_savings(y, amount, scores >= t, inv_cost) for t in grid]
    i = int(np.argmax(sav))
    return float(grid[i]), sav[i]
