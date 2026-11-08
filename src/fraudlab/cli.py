"""Command line: `python -m fraudlab.cli {data,train,evaluate}`."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import generate_claims
from .evaluate import evaluate
from .model import fit_bundle


def _load(path):
    return pd.read_csv(path, parse_dates=["claim_date"])


def _log_mlflow(bundle, report, params):
    import os

    os.environ.setdefault(
        "MLFLOW_ALLOW_FILE_STORE", "true"
    )  # newer MLflow gates the file store behind this flag
    import mlflow

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("insurance-fraud-detection")
    with mlflow.start_run():
        mlflow.log_params({**params, "threshold": round(bundle["threshold"], 4)})
        for name, m in report["models"].items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})
        mlflow.log_metrics(
            {
                "net_savings_at_threshold": report["net_savings_at_threshold"],
                "precision_at_threshold": report["precision_at_threshold"],
                "recall_at_threshold": report["recall_at_threshold"],
            }
        )
        mlflow.log_dict(report, "report.json")


def _save_report(report, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(report["monte_carlo"]["table"]).to_csv(Path(out) / "monte_carlo.csv", index=False)
    m = report["models"]
    print({k: {a: round(b, 3) for a, b in v.items()} for k, v in m.items()})
    print("threshold", round(report["threshold"], 3), report["confusion_matrix"])
    print(
        "net savings",
        round(report["net_savings_at_threshold"]),
        "of fraud loss",
        round(report["fraud_loss_without_model"]),
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="fraudlab")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--n", type=int, default=50_000)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--out", default="data/claims.csv")
    for name in ("train", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--data", default="data/claims.csv")
        s.add_argument("--model", default="models/bundle.joblib")
        s.add_argument("--reports", default="reports")
        s.add_argument("--mc-sims", type=int, default=300)
    t = sub.choices["train"]
    t.add_argument("--n-estimators", type=int, default=300)
    t.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    if a.cmd == "data":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df = generate_claims(a.n, a.seed)
        df.to_csv(a.out, index=False)
        print(f"wrote {len(df)} claims ({df.is_fraud.mean():.2%} fraud) to {a.out}")
        return
    df = _load(a.data)
    if a.cmd == "train":
        bundle = fit_bundle(df, a.n_estimators, a.seed)
        Path(a.model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, a.model)
    else:
        bundle = joblib.load(a.model)
    report = evaluate(bundle, df, mc_sims=a.mc_sims)
    if a.cmd == "train":
        _log_mlflow(bundle, report, {"n_estimators": a.n_estimators, "seed": a.seed, "rows": len(df)})
    _save_report(report, a.reports)


if __name__ == "__main__":
    main()
