"""Seeded synthetic claims generator with injected fraud patterns."""

import numpy as np
import pandas as pd

INJURIES = ["none", "soft_tissue", "fracture", "whiplash", "back", "other"]
_P_LEGIT = [0.45, 0.12, 0.08, 0.15, 0.12, 0.08]
_P_FRAUD = [0.10, 0.25, 0.05, 0.30, 0.20, 0.10]
N_GARAGES, N_PROVIDERS = 300, 150
RING_GARAGES, RING_PROVIDERS = 10, 6  # ids G0000..G0009 / P000..P005 form the fraud ring


def generate_claims(n: int = 50_000, seed: int = 42, fraud_rate: float = 0.04) -> pd.DataFrame:
    """Return `n` claims over two years. Same (n, seed, fraud_rate) gives the same frame.

    Fraud is drawn first, then each feature is drawn conditionally on it. Every
    pattern is probabilistic so the classes overlap and the task is not trivial:
    young policies, inflated and round amounts, very quick or very late reports,
    repeat claimants, a ring of garages/providers and soft-tissue injuries.
    """
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < fraud_rate

    day = np.sort(rng.integers(0, 730, n))
    claim_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(day, unit="D")

    policy_age = 1 + rng.exponential(900, n)
    young = fraud & (rng.random(n) < 0.5)
    policy_age[young] = 1 + rng.exponential(40, young.sum())

    amount = rng.lognormal(np.log(3500), 0.9, n)
    amount[fraud] = rng.lognormal(np.log(7500), 0.6, fraud.sum())
    rounded = fraud & (rng.random(n) < 0.3)
    amount[rounded] = np.round(amount[rounded] / 500) * 500

    delay = rng.exponential(6, n)
    u = rng.random(n)
    quick, late = fraud & (u < 0.35), fraud & (u >= 0.35) & (u < 0.75)
    delay[quick] = rng.uniform(0, 1, quick.sum())
    delay[late] = rng.exponential(35, late.sum())

    prior = np.where(fraud, rng.poisson(1.8, n), rng.poisson(0.6, n))

    def pick(n_ids, n_ring, p_ring_legit, p_ring_fraud, prefix, width):
        ids = rng.integers(n_ring, n_ids, n)
        in_ring = rng.random(n) < np.where(fraud, p_ring_fraud, p_ring_legit)
        ids[in_ring] = rng.integers(0, n_ring, in_ring.sum())
        return np.array([f"{prefix}{i:0{width}d}" for i in ids])

    injury = np.where(
        fraud,
        rng.choice(INJURIES, n, p=_P_FRAUD),
        rng.choice(INJURIES, n, p=_P_LEGIT),
    )

    return pd.DataFrame(
        {
            "claim_id": np.arange(n),
            "claim_date": claim_date,
            "policy_age_days": policy_age.round(1),
            "claim_amount": amount.round(2),
            "report_delay_days": delay.round(2),
            "prior_claims": prior,
            "claimant_age": np.clip(rng.normal(42, 14, n), 18, 85).round().astype(int),
            "garage_id": pick(N_GARAGES, RING_GARAGES, 0.02, 0.35, "G", 4),
            "provider_id": pick(N_PROVIDERS, RING_PROVIDERS, 0.02, 0.25, "P", 3),
            "injury_type": injury,
            "is_fraud": fraud.astype(int),
        }
    )
