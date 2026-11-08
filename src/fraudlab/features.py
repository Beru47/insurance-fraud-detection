"""Feature engineering.

Garage/provider fraud rates are smoothed target encodings, fitted on training data only
and computed out-of-fold on the training set to avoid leakage."""

import numpy as np
import pandas as pd

from .data import INJURIES

ID_COLS = ("garage_id", "provider_id")
NUMERIC = ["claim_amount", "policy_age_days", "report_delay_days", "prior_claims", "claimant_age"]


class RateEncoder:
    """Smoothed fraud rate per id: (frauds + m*prior) / (claims + m)."""

    def __init__(self, smoothing: float = 20.0):
        self.m = smoothing

    def _rates(self, df, col):
        g = df.groupby(col)["is_fraud"].agg(["sum", "count"])
        return ((g["sum"] + self.m * self.prior) / (g["count"] + self.m)).to_dict()

    def fit(self, df: pd.DataFrame) -> "RateEncoder":
        self.prior = float(df["is_fraud"].mean())
        self.rates = {c: self._rates(df, c) for c in ID_COLS}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {f"{c}_rate": df[c].map(self.rates[c]).fillna(self.prior) for c in ID_COLS},
            index=df.index,
        )

    def oof(self, df: pd.DataFrame, k: int = 5, seed: int = 0) -> pd.DataFrame:
        """Encode each fold with rates learned on the other folds."""
        folds = np.random.default_rng(seed).integers(0, k, len(df))
        out = pd.DataFrame(index=df.index, columns=[f"{c}_rate" for c in ID_COLS], dtype=float)
        for f in range(k):
            tr, te = df[folds != f], df[folds == f]
            for c in ID_COLS:
                out.loc[te.index, f"{c}_rate"] = te[c].map(self._rates(tr, c)).fillna(self.prior)
        return out


def build_features(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC].astype(float).copy()
    X["log_amount"] = np.log1p(X["claim_amount"])
    X["amount_per_policy_day"] = np.log1p(X["claim_amount"] / (X["policy_age_days"] + 1))
    X["new_policy"] = (X["policy_age_days"] < 30).astype(float)
    X["quick_report"] = (X["report_delay_days"] < 1).astype(float)
    X["round_amount"] = (df["claim_amount"] % 500 == 0).astype(float)
    inj = pd.get_dummies(pd.Categorical(df["injury_type"], categories=INJURIES), prefix="injury", dtype=float)
    inj.index = df.index
    return pd.concat([X, inj, rates.astype(float)], axis=1)
