from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


def td_to_seconds(x) -> float:
    if pd.isna(x):
        return np.nan
    try:
        return pd.to_timedelta(x).total_seconds()
    except Exception:
        return np.nan


def convert_time_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(td_to_seconds).astype(float)
    return df


def normalize_paths(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["path"] = out["path"].astype(str)
    out["path"] = out["path"].str.replace("\\\\", "/", regex=False).str.replace("\\", "/", regex=False)
    return out


def to_bool_series(x: pd.Series) -> pd.Series:
    if not isinstance(x, pd.Series):
        return pd.Series([], dtype=bool)
    return (
        x.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True, "1": True, "t": True, "yes": True, "y": True,
            "false": False, "0": False, "f": False, "no": False, "n": False
        })
        .fillna(False)
    )


def valid_quali_mask(df: pd.DataFrame) -> pd.Series:
    if "LapTime" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    del_flags = to_bool_series(df.get("Deleted", pd.Series(False, index=df.index)))
    m = (~del_flags) & df["LapTime"].notna()
    m &= (df["LapTime"] >= 30) & (df["LapTime"] <= 200)
    return m


def _between_iqr_bounds(s: pd.Series) -> pd.Series:
    s_valid = s.dropna()
    if len(s_valid) < 3:
        return s.between(40, 240, inclusive="both")
    q1, q3 = np.nanpercentile(s_valid, [25, 75])
    iqr = q3 - q1
    lo = max(40, q1 - 1.5 * iqr)
    hi = min(240, q3 + 1.5 * iqr)
    return s.between(lo, hi, inclusive="both")


def valid_race_mask_iqr(df: pd.DataFrame) -> pd.Series:
    if "LapTime" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    del_flags = to_bool_series(df.get("Deleted", pd.Series(False, index=df.index)))
    m = (~del_flags) & df["LapTime"].notna()
    if "Abbreviation" in df.columns and len(df):
        bounds = df.groupby("Abbreviation", group_keys=False)["LapTime"].apply(_between_iqr_bounds)
        bounds = bounds.reindex(df.index)
        m &= bounds.fillna(False)
    else:
        m &= _between_iqr_bounds(df["LapTime"]).fillna(False)
    return m


def pace_summaries(lap_seconds: pd.Series) -> Dict[str, float]:
    x = lap_seconds.dropna().to_numpy()
    if x.size == 0:
        return dict(best=np.nan, median=np.nan, std=np.nan, p95=np.nan, top3=np.nan)
    x_sorted = np.sort(x)
    best = float(x_sorted[0])
    top3 = float(np.mean(x_sorted[: min(3, x_sorted.size)]))
    return dict(
        best=best,
        median=float(np.median(x)),
        std=float(np.std(x, ddof=1) if x.size > 1 else 0.0),
        p95=float(np.percentile(x, 95)),
        top3=top3,
    )
