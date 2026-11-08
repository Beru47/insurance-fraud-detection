import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudlab import serve
from fraudlab.costs import best_threshold, net_savings
from fraudlab.data import generate_claims
from fraudlab.evaluate import recall_at_precision
from fraudlab.features import RateEncoder, build_features
from fraudlab.model import fit_bundle, time_split
from fraudlab.montecarlo import simulate

CLAIM = {
    "claim_amount": 9000,
    "policy_age_days": 12,
    "report_delay_days": 0.5,
    "prior_claims": 3,
    "claimant_age": 35,
    "garage_id": "G0002",
    "provider_id": "P001",
    "injury_type": "whiplash",
}


@pytest.fixture(scope="module")
def df():
    return generate_claims(6000, seed=1)


@pytest.fixture(scope="module")
def bundle(df):
    return fit_bundle(df, n_estimators=30)


def test_generator_is_deterministic_and_fraud_rate_in_range():
    a, b = generate_claims(3000, seed=7), generate_claims(3000, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(generate_claims(3000, seed=8))
    assert 0.03 <= generate_claims(20000, seed=7).is_fraud.mean() <= 0.05


def test_time_split_is_chronological(df):
    tr, va, te = time_split(df)
    assert tr.claim_date.max() <= va.claim_date.min() <= va.claim_date.max() <= te.claim_date.min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_features_no_nans_and_oof_differs_from_in_sample(df):
    tr, _, _ = time_split(df)
    enc = RateEncoder().fit(tr)
    X = build_features(tr, enc.oof(tr))
    assert not X.isna().any().any()
    # out-of-fold rates must not equal the in-sample ones (that would be target leakage)
    assert not np.allclose(X["garage_id_rate"], enc.transform(tr)["garage_id_rate"])
    unseen = build_features(
        df.head(1).assign(garage_id="G9999"), enc.transform(df.head(1).assign(garage_id="G9999"))
    )
    assert unseen["garage_id_rate"].iloc[0] == pytest.approx(enc.prior)


def test_cost_threshold_logic():
    y = np.array([1, 1, 0, 0, 0, 0])
    amount = np.array([1000.0, 50.0, 500, 500, 500, 500])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.1])
    # flagging the 50-euro fraud costs 200 to investigate: not worth it, so the cut sits above 0.8
    t, s = best_threshold(scores, y, amount, inv_cost=200, grid=[0.05, 0.75, 0.85])
    assert t == 0.85 and s == pytest.approx(800)
    assert net_savings(y, amount, scores >= 0.75, 200) == pytest.approx(1050 - 400)
    # free investigations: flag everything fraudulent
    assert best_threshold(scores, y, amount, inv_cost=0, grid=[0.75, 0.85, 0.95])[0] == 0.75


def test_recall_at_precision():
    y = np.array([1, 0, 1, 0])
    assert recall_at_precision(y, np.array([0.9, 0.8, 0.7, 0.1]), 1.0) == pytest.approx(0.5)


def test_monte_carlo_monotone_and_reproducible():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    scores = np.clip(y * 0.6 + rng.random(2000) * 0.4, 0, 1)
    amount = np.full(2000, 5000.0)
    a = simulate(scores, y, amount, [0.1, 0.5, 0.9], n_claims=2000, n_sims=20, seed=3)
    b = simulate(scores, y, amount, [0.1, 0.5, 0.9], n_claims=2000, n_sims=20, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a.savings_mean.iloc[1] > a.savings_mean.iloc[0]  # flagging everything wastes money
    assert (a.savings_p5 <= a.savings_mean).all() and (a.savings_mean <= a.savings_p95).all()


def test_api_and_lambda_share_one_predictor(bundle, monkeypatch):
    monkeypatch.setattr(serve, "get_predictor", lambda: serve.Predictor(bundle))
    r = TestClient(serve.app).post("/score", json=CLAIM)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_score"] <= 1 and body["threshold"] == bundle["threshold"]
    ev = serve.handler({"body": json.dumps(CLAIM)}, None)
    assert ev["statusCode"] == 200 and json.loads(ev["body"]) == body
    assert serve.handler({"claim_amount": -1}, None)["statusCode"] == 400
    assert TestClient(serve.app).post("/score", json={**CLAIM, "claim_amount": -5}).status_code == 422


def test_predictor_loaded_once(bundle, tmp_path, monkeypatch):
    import joblib

    path = tmp_path / "b.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setattr(serve, "MODEL_PATH", str(path))
    calls = []
    orig = serve.Predictor.load
    monkeypatch.setattr(
        serve.Predictor,
        "load",
        classmethod(lambda cls, p=str(path): calls.append(1) or orig.__func__(cls, p)),
    )
    serve.get_predictor.cache_clear()
    assert serve.get_predictor() is serve.get_predictor()
    assert len(calls) == 1
    serve.get_predictor.cache_clear()
