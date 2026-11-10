"""End-to-end unit and integration tests for the Fraud Risk Engine with XAI."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.figure
import pandas as pd

from fraud_risk_engine.app import build_interface
from fraud_risk_engine.data_generator import generate_synthetic_transactions
from fraud_risk_engine.explainer import FraudExplainer
from fraud_risk_engine.features import FEATURE_COLUMNS, engineer_features
from fraud_risk_engine.train import train_and_save


class TestFeatureEngineering(unittest.TestCase):
    def test_rolling_velocity_and_geospatial_distance(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "cc_num": 101,
                    "trans_date_trans_time": "2026-02-01 10:00:00",
                    "amt": 50.0,
                    "category": "Grocery",
                    "lat": 10.0,
                    "long": 20.0,
                    "merch_lat": 13.0,
                    "merch_long": 24.0,
                    "is_fraud": 0,
                },
                {
                    "cc_num": 101,
                    "trans_date_trans_time": "2026-02-01 10:00:30",
                    "amt": 200.0,
                    "category": "Shopping",
                    "lat": 10.0,
                    "long": 20.0,
                    "merch_lat": 10.0,
                    "merch_long": 20.0,
                    "is_fraud": 1,
                },
            ]
        )

        engineered, cat_map = engineer_features(raw)
        self.assertEqual(set(FEATURE_COLUMNS).issubset(engineered.columns), True)
        self.assertEqual(engineered.loc[0, "Seconds_Since_Last_Txn"], 999999.0)
        self.assertEqual(engineered.loc[1, "Seconds_Since_Last_Txn"], 30.0)
        self.assertEqual(engineered.loc[1, "Txn_Velocity_24h"], 2.0)
        self.assertEqual(engineered.loc[1, "Total_Spent_24h"], 250.0)
        self.assertAlmostEqual(engineered.loc[0, "Distance_to_Merchant"], 5.0, places=5)
        self.assertEqual(cat_map["Grocery"], 1)
        self.assertEqual(cat_map["Shopping"], 2)

    def test_missing_columns_raises(self) -> None:
        bad = pd.DataFrame([{"amt": 10.0}])
        with self.assertRaises(ValueError):
            engineer_features(bad)


class TestTrainingExplainerAndGradio(unittest.TestCase):
    def test_train_save_explain_and_gradio_interface(self) -> None:
        df = generate_synthetic_transactions(n_rows=600, fraud_rate=0.05, seed=7)
        self.assertEqual(len(df), 600)
        self.assertGreater(int(df["is_fraud"].sum()), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = train_and_save(
                output_dir=tmpdir,
                synthetic_rows=1200,
                fraud_rate=0.03,
                n_estimators=40,
                random_state=42,
            )
            model_file = Path(out["model_path"])
            meta_file = Path(out["metadata_path"])
            self.assertTrue(model_file.exists())
            self.assertTrue(meta_file.exists())

            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self.assertEqual(meta["feature_columns"], FEATURE_COLUMNS)
            self.assertGreater(meta["metrics"]["roc_auc"], 0.85)

            explainer = FraudExplainer.from_artifacts(tmpdir)

            # High-velocity late-night distant spend -> expected higher risk score
            high_risk = explainer.assess(
                amount=1200.0,
                category="Shopping",
                seconds_since_last=5.0,
                velocity_24h=12,
                spent_24h=3500.0,
                hour=2,
                distance=18.0,
            )
            # Low-amount daytime near-home transaction -> expected approved
            low_risk = explainer.assess(
                amount=22.0,
                category="Grocery",
                seconds_since_last=80000.0,
                velocity_24h=1,
                spent_24h=22.0,
                hour=14,
                distance=0.1,
            )

            self.assertGreater(high_risk.risk_score_pct, low_risk.risk_score_pct)
            self.assertTrue(high_risk.blocked)
            self.assertFalse(low_risk.blocked)

            verdict_text, fig = explainer.predict_and_explain(
                amount=1200.0,
                category="Shopping",
                seconds_since_last=5.0,
                velocity_24h=12,
                spent_24h=3500.0,
                hour=2,
                distance=18.0,
            )
            self.assertIn("TRANSACTION BLOCKED", verdict_text)
            self.assertIsInstance(fig, matplotlib.figure.Figure)

            demo = build_interface(explainer)
            self.assertEqual(len(demo.input_components), 7)
            self.assertEqual(len(demo.output_components), 2)
            # Verify bound Gradio callback runs cleanly
            cb_verdict, cb_fig = demo.fn(22.0, "Grocery", 80000.0, 1, 22.0, 14, 0.1)
            self.assertIn("TRANSACTION APPROVED", cb_verdict)
            self.assertIsInstance(cb_fig, matplotlib.figure.Figure)


if __name__ == "__main__":
    unittest.main()
