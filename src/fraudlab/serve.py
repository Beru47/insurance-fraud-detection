"""Serving: one Predictor shared by a FastAPI app and an AWS Lambda-style handler.

The model bundle is loaded once per process (lru_cache), so warm Lambda invocations
and API requests reuse it."""

import json
import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .data import INJURIES
from .features import build_features
from .model import score_features

MODEL_PATH = os.environ.get("MODEL_PATH", "models/bundle.joblib")


class Claim(BaseModel):
    claim_amount: float = Field(gt=0)
    policy_age_days: float = Field(ge=0)
    report_delay_days: float = Field(ge=0)
    prior_claims: int = Field(ge=0)
    claimant_age: int = Field(ge=16, le=110)
    garage_id: str
    provider_id: str
    injury_type: str = Field(json_schema_extra={"enum": INJURIES})


class Predictor:
    def __init__(self, bundle: dict):
        self.b = bundle

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "Predictor":
        # the bundle is produced by our own `train` step; never load untrusted files
        return cls(joblib.load(path))

    def score(self, claim: dict) -> dict:
        df = pd.DataFrame([claim])
        X = build_features(df, self.b["encoder"].transform(df))[self.b["columns"]]
        s = float(score_features(self.b["models"], X)["ensemble"][0])
        return {
            "fraud_score": round(s, 4),
            "flagged": s >= self.b["threshold"],
            "threshold": self.b["threshold"],
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor.load()


app = FastAPI(title="insurance-fraud-detection")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(claim: Claim):
    return get_predictor().score(claim.model_dump())


def handler(event, context=None):
    """Lambda entry point. `event` is a claim dict or an API Gateway event with a JSON `body`."""
    try:
        body = event.get("body", event) if isinstance(event, dict) else event
        claim = Claim(**(json.loads(body) if isinstance(body, str) else body))
        return {"statusCode": 200, "body": json.dumps(get_predictor().score(claim.model_dump()))}
    except Exception as e:  # invalid payloads return 400 instead of crashing the invocation
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}
