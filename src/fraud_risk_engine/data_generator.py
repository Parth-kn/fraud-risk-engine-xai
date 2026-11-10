"""Dataset loading (Kaggle CSV) and realistic synthetic transaction generator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .features import DEFAULT_CATEGORIES


def load_kaggle_transactions(csv_path: str | Path, max_rows: Optional[int] = 200_000) -> pd.DataFrame:
    """Load Kaggle fraudTrain.csv (kartik2112/fraud-detection) safely."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")
    return pd.read_csv(path, nrows=max_rows)


def generate_synthetic_transactions(
    n_rows: int = 8000,
    fraud_rate: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic imbalanced card transactions for offline training and testing.

    Simulates legitimate user baselines vs. fraudulent midnight velocity spikes
    and anomalous merchant distance patterns mirroring the real Fraud Risk problem.
    """
    if n_rows < 50:
        raise ValueError("n_rows must be >= 50 for meaningful stratified splits.")
    if not (0.002 <= fraud_rate <= 0.4):
        raise ValueError("fraud_rate must be between 0.002 and 0.4.")

    rng = np.random.default_rng(seed)
    n_cards = max(20, n_rows // 40)
    card_ids = np.arange(4_000_000_000_000_000, 4_000_000_000_000_000 + n_cards)

    home_lats = rng.uniform(12.0, 42.0, size=n_cards)
    home_longs = rng.uniform(72.0, 88.0, size=n_cards)

    n_fraud = max(10, int(round(n_rows * fraud_rate)))
    n_legit = n_rows - n_fraud

    base_ts = pd.Timestamp("2026-01-01 00:00:00")

    # Legitimate baseline transactions (daytime bias, small distance, normal cadence)
    legit_card_idx = rng.integers(0, n_cards, size=n_legit)
    legit_hours = rng.choice(np.arange(7, 23), size=n_legit)
    legit_day_offsets = rng.integers(0, 30, size=n_legit)
    legit_times = [
        base_ts + pd.Timedelta(days=int(d), hours=int(h), minutes=int(m))
        for d, h, m in zip(legit_day_offsets, legit_hours, rng.integers(0, 60, size=n_legit))
    ]
    legit_df = pd.DataFrame(
        {
            "cc_num": card_ids[legit_card_idx],
            "trans_date_trans_time": legit_times,
            "amt": np.round(rng.lognormal(mean=3.5, sigma=0.7, size=n_legit), 2),
            "category": rng.choice(DEFAULT_CATEGORIES, size=n_legit),
            "lat": home_lats[legit_card_idx],
            "long": home_longs[legit_card_idx],
            "merch_lat": home_lats[legit_card_idx] + rng.normal(0.0, 0.2, size=n_legit),
            "merch_long": home_longs[legit_card_idx] + rng.normal(0.0, 0.2, size=n_legit),
            "is_fraud": 0,
        }
    )

    # Fraudulent bursts (clustered swipes within seconds, late-night hours, distant merchants)
    fraud_rows = []
    victim_pool = rng.integers(0, n_cards, size=max(5, n_fraud // 3))
    generated = 0
    while generated < n_fraud:
        v_idx = int(rng.choice(victim_pool))
        burst_len = min(int(rng.integers(2, 5)), n_fraud - generated)
        burst_day = int(rng.integers(1, 29))
        burst_hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
        burst_start = base_ts + pd.Timedelta(
            days=burst_day, hours=burst_hour, minutes=int(rng.integers(0, 50))
        )
        for j in range(burst_len):
            t_stamp = burst_start + pd.Timedelta(seconds=j * int(rng.integers(3, 40)))
            fraud_rows.append(
                {
                    "cc_num": int(card_ids[v_idx]),
                    "trans_date_trans_time": t_stamp,
                    "amt": float(np.round(rng.uniform(250.0, 1800.0), 2)),
                    "category": str(rng.choice(["Shopping", "Travel", "Entertainment"])),
                    "lat": float(home_lats[v_idx]),
                    "long": float(home_longs[v_idx]),
                    "merch_lat": float(home_lats[v_idx] + rng.choice([-1, 1]) * rng.uniform(5.0, 20.0)),
                    "merch_long": float(home_longs[v_idx] + rng.choice([-1, 1]) * rng.uniform(5.0, 20.0)),
                    "is_fraud": 1,
                }
            )
            generated += 1

    combined = pd.concat([legit_df, pd.DataFrame(fraud_rows)], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
