"""Training CLI and evaluation pipeline for the XGBoost fraud risk classifier."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .data_generator import generate_synthetic_transactions, load_kaggle_transactions
from .features import FEATURE_COLUMNS, engineer_features

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


@dataclass
class TrainingMetrics:
    recall: float
    precision: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    scale_pos_weight: float
    train_rows: int
    test_rows: int
    fraud_test_count: int


def train_and_save(
    output_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    csv_path: Optional[str | Path] = None,
    max_rows: int = 200_000,
    synthetic_rows: int = 8000,
    fraud_rate: float = 0.015,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Dict[str, object]:
    """Engineer features, train XGBClassifier with imbalance weighting, and save artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if csv_path:
        raw_df = load_kaggle_transactions(csv_path, max_rows=max_rows)
        dataset_source = str(csv_path)
    else:
        raw_df = generate_synthetic_transactions(
            n_rows=synthetic_rows, fraud_rate=fraud_rate, seed=random_state
        )
        dataset_source = f"synthetic(rows={synthetic_rows}, fraud_rate={fraud_rate})"

    engineered_df, category_map = engineer_features(raw_df)
    X = engineered_df[FEATURE_COLUMNS]
    y = engineered_df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    scale_pos_weight = neg_count / max(pos_count, 1.0)

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=7,
        learning_rate=0.05,
        n_estimators=n_estimators,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = TrainingMetrics(
        recall=round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        precision=round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        f1_score=round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        roc_auc=round(float(roc_auc_score(y_test, y_prob)), 4),
        pr_auc=round(float(average_precision_score(y_test, y_prob)), 4),
        scale_pos_weight=round(float(scale_pos_weight), 2),
        train_rows=int(len(X_train)),
        test_rows=int(len(X_test)),
        fraud_test_count=int(np.sum(y_test == 1)),
    )

    model_file = out_path / "fraud_xgboost_model.json"
    meta_file = out_path / "model_metadata.json"

    model.save_model(str(model_file))
    metadata = {
        "dataset_source": dataset_source,
        "feature_columns": FEATURE_COLUMNS,
        "category_map": category_map,
        "threshold_pct": 65.0,
        "metrics": asdict(metrics),
    }
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_file),
        "metadata_path": str(meta_file),
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Credit Card Fraud Risk Engine (XGBoost)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to Kaggle fraudTrain.csv")
    parser.add_argument("--max-rows", type=int, default=200_000, help="Max CSV rows to load")
    parser.add_argument("--synthetic-rows", type=int, default=8000, help="Synthetic rows when no CSV given")
    parser.add_argument("--n-estimators", type=int, default=200, help="XGBoost trees count")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR), help="Artifacts folder")
    args = parser.parse_args()

    result = train_and_save(
        output_dir=args.output_dir,
        csv_path=args.dataset,
        max_rows=args.max_rows,
        synthetic_rows=args.synthetic_rows,
        n_estimators=args.n_estimators,
    )
    print(json.dumps(result["metadata"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
