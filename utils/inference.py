from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _mmssms_to_seconds(s: str | float | int | None) -> float | np.nan:
    """Parse strings like '1:18.792' -> 78.792 seconds. Empty -> NaN."""
    if s is None:
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return np.nan
    if ":" not in s:
        try:
            return float(s)
        except Exception:
            return np.nan
    mm, ss = s.split(":", 1)
    try:
        return int(mm) * 60 + float(ss)
    except Exception:
        return np.nan


def add_quali_seconds(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Q1", "Q2", "Q3"]:
        if col in df.columns:
            df[f"{col.lower()}_s"] = df[col].apply(_mmssms_to_seconds)
    q_cols = [c for c in ["q1_s", "q2_s", "q3_s"] if c in df.columns]
    if q_cols:
        df["quali_best_s"] = df[q_cols].min(axis=1)
    return df




def align_to_training_columns(
    raw: pd.DataFrame,
    expected_num: list[str],
    expected_cat: list[str],
) -> pd.DataFrame:
    """Ensure raw has all columns used during training (order & presence).
    Missing numeric -> NaN; missing categoricals -> 'missing'. Extra cols are dropped.
    """
    df = raw.copy()

    
    for c in expected_num:
        if c not in df.columns:
            df[c] = np.nan
    for c in expected_cat:
        if c not in df.columns:
            df[c] = "missing"

    df = df[expected_cat + expected_num]  # ColumnTransformer used (cat, num)
    return df