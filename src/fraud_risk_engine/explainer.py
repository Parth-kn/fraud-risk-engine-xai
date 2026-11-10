"""Real-time fraud risk scoring and SHAP waterfall explainability engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb

from .features import DEFAULT_CATEGORY_MAP, FEATURE_COLUMNS


@dataclass
class FraudAssessment:
    risk_score_pct: float
    blocked: bool
    verdict_text: str
    feature_frame: pd.DataFrame


class FraudExplainer:
    """Wraps a trained XGBoost risk classifier and SHAP TreeExplainer."""

    def __init__(
        self,
        model: xgb.XGBClassifier,
        category_map: Dict[str, int] | None = None,
        threshold_pct: float = 65.0,
    ) -> None:
        self.model = model
        self.category_map = category_map or dict(DEFAULT_CATEGORY_MAP)
        self.threshold_pct = float(threshold_pct)
        self.explainer = shap.TreeExplainer(self.model)

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path) -> "FraudExplainer":
        """Load persisted XGBoost model JSON and category metadata."""
        base = Path(artifact_dir)
        model_path = base / "fraud_xgboost_model.json"
        meta_path = base / "model_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = xgb.XGBClassifier()
        model.load_model(str(model_path))

        category_map = dict(DEFAULT_CATEGORY_MAP)
        threshold_pct = 65.0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            category_map = meta.get("category_map", category_map)
            threshold_pct = float(meta.get("threshold_pct", threshold_pct))

        return cls(model=model, category_map=category_map, threshold_pct=threshold_pct)

    def build_input_frame(
        self,
        amount: float,
        category: str,
        seconds_since_last: float,
        velocity_24h: float,
        spent_24h: float,
        hour: int,
        distance: float,
    ) -> pd.DataFrame:
        """Assemble single-transaction feature frame aligned to training columns."""
        cat_code = self.category_map.get(str(category).strip(), 0)
        frame = pd.DataFrame(
            {
                "amt": [float(amount)],
                "category_code": [int(cat_code)],
                "Seconds_Since_Last_Txn": [float(seconds_since_last)],
                "Txn_Velocity_24h": [float(velocity_24h)],
                "Total_Spent_24h": [float(spent_24h)],
                "Transaction_Hour": [int(hour)],
                "Distance_to_Merchant": [float(distance)],
            }
        )
        return frame[FEATURE_COLUMNS]

    def assess(
        self,
        amount: float,
        category: str,
        seconds_since_last: float,
        velocity_24h: float,
        spent_24h: float,
        hour: int,
        distance: float,
    ) -> FraudAssessment:
        """Score a transaction and return verdict text alongside engineered inputs."""
        input_df = self.build_input_frame(
            amount=amount,
            category=category,
            seconds_since_last=seconds_since_last,
            velocity_24h=velocity_24h,
            spent_24h=spent_24h,
            hour=hour,
            distance=distance,
        )
        risk_score_pct = float(self.model.predict_proba(input_df)[0][1] * 100.0)
        blocked = risk_score_pct > self.threshold_pct
        if blocked:
            verdict = (
                f"TRANSACTION BLOCKED\n"
                f"Risk Confidence: {risk_score_pct:.2f}%\n"
                f"Reason: High Behavioral/Geospatial Anomaly Detected."
            )
        else:
            verdict = (
                f"TRANSACTION APPROVED\n"
                f"Risk Confidence: {risk_score_pct:.2f}%\n"
                f"Status: Normal Behavioral Baseline."
            )
        return FraudAssessment(
            risk_score_pct=risk_score_pct,
            blocked=blocked,
            verdict_text=verdict,
            feature_frame=input_df,
        )

    def predict_and_explain(
        self,
        amount: float,
        category: str,
        seconds_since_last: float,
        velocity_24h: float,
        spent_24h: float,
        hour: int,
        distance: float,
    ) -> Tuple[str, plt.Figure | None]:
        """Compute risk score verdict and render a SHAP waterfall plot."""
        try:
            assessment = self.assess(
                amount=amount,
                category=category,
                seconds_since_last=seconds_since_last,
                velocity_24h=velocity_24h,
                spent_24h=spent_24h,
                hour=hour,
                distance=distance,
            )
            plt.close("all")
            shap_values = self.explainer(assessment.feature_frame)
            fig = plt.figure(figsize=(8, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            return assessment.verdict_text, fig
        except Exception as exc:
            return f"SYSTEM ERROR:\n{exc}", None
