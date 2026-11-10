"""Behavioral velocity, temporal, and geospatial feature engineering for fraud detection."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amt",
    "category_code",
    "Seconds_Since_Last_Txn",
    "Txn_Velocity_24h",
    "Total_Spent_24h",
    "Transaction_Hour",
    "Distance_to_Merchant",
]

DEFAULT_CATEGORIES = [
    "Grocery",
    "Shopping",
    "Travel",
    "Entertainment",
    "Gas/Transport",
]

DEFAULT_CATEGORY_MAP: Dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(DEFAULT_CATEGORIES)
}


def build_category_mapping(categories: Iterable[str]) -> Dict[str, int]:
    """Build a deterministic 1-indexed mapping (0 reserved for unseen categories)."""
    unique = sorted({str(c).strip() for c in categories if pd.notna(c) and str(c).strip()})
    return {cat: idx + 1 for idx, cat in enumerate(unique)}


def engineer_features(
    raw_df: pd.DataFrame,
    category_map: Dict[str, int] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Transform raw card transactions into engineered velocity and geospatial features.

    Args:
        raw_df: Transaction DataFrame containing ``cc_num``, ``trans_date_trans_time``,
            ``amt``, ``category``, ``lat``, ``long``, ``merch_lat``, and ``merch_long``.
        category_map: Optional precomputed category-to-integer map for inference parity.

    Returns:
        Tuple of ``(engineered_df, used_category_map)``.
    """
    required_cols = {
        "cc_num",
        "trans_date_trans_time",
        "amt",
        "category",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
    }
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {sorted(missing)}")

    df = raw_df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="raise")
    df = df.sort_values(by=["cc_num", "trans_date_trans_time"], kind="mergesort").reset_index(drop=True)

    # Inter-transaction delta detects high-speed automated/bot swipe bursts.
    df["Seconds_Since_Last_Txn"] = (
        df.groupby("cc_num", sort=False)["trans_date_trans_time"]
        .diff()
        .dt.total_seconds()
        .fillna(999999.0)
        .clip(lower=0.0)
    )

    # Compute 24-hour rolling velocity and cumulative spend per cardholder.
    rolled = (
        df.set_index("trans_date_trans_time")
        .groupby("cc_num", sort=False)["amt"]
        .rolling("24h")
    )
    df["Txn_Velocity_24h"] = rolled.count().to_numpy(dtype=float)
    df["Total_Spent_24h"] = rolled.sum().to_numpy(dtype=float)

    df["Transaction_Hour"] = df["trans_date_trans_time"].dt.hour.astype(int)
    df["Distance_to_Merchant"] = np.sqrt(
        (df["lat"].astype(float) - df["merch_lat"].astype(float)) ** 2
        + (df["long"].astype(float) - df["merch_long"].astype(float)) ** 2
    )

    resolved_map = category_map or build_category_mapping(df["category"])
    df["category_code"] = (
        df["category"].astype(str).str.strip().map(resolved_map).fillna(0).astype(int)
    )

    return df, resolved_map
