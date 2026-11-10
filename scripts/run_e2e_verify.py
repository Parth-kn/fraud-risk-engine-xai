"""Non-interactive end-to-end verification for training, SHAP plots, and live Gradio UI."""

from __future__ import annotations

import json
import socket
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_risk_engine.app import build_interface
from fraud_risk_engine.explainer import FraudExplainer
from fraud_risk_engine.train import train_and_save


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print("1. Training synthetic imbalanced XGBoost model...")
        res = train_and_save(
            output_dir=tmp_path,
            synthetic_rows=4000,
            fraud_rate=0.02,
            n_estimators=80,
        )
        print("   Metrics:", json.dumps(res["metadata"]["metrics"]))

        print("2. Loading FraudExplainer & generating SHAP waterfall figure...")
        explainer = FraudExplainer.from_artifacts(tmp_path)
        verdict, fig = explainer.predict_and_explain(
            amount=1450.0,
            category="Shopping",
            seconds_since_last=3.0,
            velocity_24h=14,
            spent_24h=4200.0,
            hour=2,
            distance=16.5,
        )
        assert fig is not None, "Expected SHAP matplotlib figure"
        plot_file = tmp_path / "shap_waterfall_test.png"
        fig.savefig(plot_file, dpi=120)
        assert plot_file.exists() and plot_file.stat().st_size > 1000
        print("   Verdict:", verdict.replace("\n", " | "))
        print(f"   Saved SHAP PNG verification ({plot_file.stat().st_size} bytes)")

        print("3. Launching Gradio interface locally (prevent_thread_lock=True)...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        demo = build_interface(explainer)
        _, local_url, _ = demo.launch(
            server_name="127.0.0.1",
            server_port=free_port,
            prevent_thread_lock=True,
            quiet=True,
        )
        try:
            with urllib.request.urlopen(local_url, timeout=8) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                assert resp.status == 200
                assert "Fraud Risk Engine" in body or "gradio" in body.lower()
                print(f"   Gradio HTTP {resp.status} verified at {local_url} (len={len(body)})")
        finally:
            demo.close()

    print("ALL E2E CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
