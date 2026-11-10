"""Gradio interactive monitoring interface for real-time fraud scoring & XAI."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from .explainer import FraudExplainer
from .features import DEFAULT_CATEGORIES
from .train import DEFAULT_ARTIFACT_DIR, train_and_save


def get_or_create_explainer(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> FraudExplainer:
    """Load saved artifacts or train a fast baseline model on synthetic data if absent."""
    base = Path(artifact_dir)
    if not (base / "fraud_xgboost_model.json").exists():
        train_and_save(output_dir=base, synthetic_rows=4000, n_estimators=100)
    return FraudExplainer.from_artifacts(base)


def build_interface(explainer: FraudExplainer | None = None) -> gr.Interface:
    """Construct the Gradio dashboard without starting the HTTP server."""
    active_explainer = explainer or get_or_create_explainer()
    categories = sorted(active_explainer.category_map.keys()) or list(DEFAULT_CATEGORIES)

    return gr.Interface(
        fn=active_explainer.predict_and_explain,
        title="Intelligent Real-Time Fraud Risk Engine with XAI",
        description=(
            "Evaluates credit card fraud risk using behavioral swipe velocity, rolling 24h spend, "
            "temporal signals, and merchant distance. The SHAP chart proves which features drove the verdict."
        ),
        inputs=[
            gr.Number(label="Transaction Amount ($)", value=800.00),
            gr.Dropdown(categories, label="Merchant Category", value=categories[0]),
            gr.Slider(minimum=1, maximum=86400, step=10, label="Seconds Since Last Swipe", value=2),
            gr.Slider(minimum=1, maximum=50, step=1, label="Swipes in Last 24 Hours (Velocity)", value=15),
            gr.Number(label="Total Money Spent in Last 24 Hours ($)", value=2500.00),
            gr.Slider(minimum=0, maximum=23, step=1, label="Hour of the Day (0 = Midnight)", value=3),
            gr.Number(label="Distance to Merchant (Euclidean Units)", value=15.0),
        ],
        outputs=[
            gr.Textbox(label="AI Risk Engine Verdict", lines=3),
            gr.Plot(label="SHAP Explainable AI (XAI) Analysis"),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Fraud Risk Engine Gradio Web UI")
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    explainer = get_or_create_explainer(args.artifact_dir)
    demo = build_interface(explainer)
    demo.launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
