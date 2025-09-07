from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from utils.common import td_to_seconds, to_bool_series


def _agg_numeric(x: pd.Series, name: str) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce")
    out = {
        f"{name}_mean": float(np.nanmean(x)) if x.notna().any() else np.nan,
        f"{name}_std":  float(np.nanstd(x, ddof=1)) if x.notna().sum() > 1 else 0.0,
        f"{name}_min":  float(np.nanmin(x)) if x.notna().any() else np.nan,
        f"{name}_max":  float(np.nanmax(x)) if x.notna().any() else np.nan,
    }
    return out


def _wind_direction_features(deg: pd.Series) -> Dict[str, float]:
    d = pd.to_numeric(deg, errors="coerce")
    rad = np.deg2rad(d)
    sin_m = float(np.nanmean(np.sin(rad))) if d.notna().any() else np.nan
    cos_m = float(np.nanmean(np.cos(rad))) if d.notna().any() else np.nan
    if np.isnan(sin_m) or np.isnan(cos_m):
        mean_deg = np.nan
    else:
        mean_deg = float((np.degrees(np.arctan2(sin_m, cos_m)) + 360.0) % 360.0)
    return {
        "wind_dir_x_mean": cos_m,  
        "wind_dir_y_mean": sin_m,  
        "wind_dir_deg_mean": mean_deg,
    }


def aggregate_weather_session(wx: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize a session's weather.csv into one row.
    Columns expected (case-sensitive from your example):
      Time, AirTemp, Humidity, Pressure, Rainfall, TrackTemp, WindDirection, WindSpeed
    Returns a single-row DataFrame without any session prefix.
    """
    if wx is None or wx.empty:
        cols = [
            "wx_samples", "wx_span_s", "rain_rate", "rain_any",
            "air_temp_mean","air_temp_std","air_temp_min","air_temp_max",
            "track_temp_mean","track_temp_std","track_temp_min","track_temp_max",
            "humidity_mean","humidity_std","humidity_min","humidity_max",
            "pressure_mean","pressure_std","pressure_min","pressure_max",
            "wind_speed_mean","wind_speed_std","wind_speed_min","wind_speed_max",
            "wind_dir_x_mean","wind_dir_y_mean","wind_dir_deg_mean",
        ]
        return pd.DataFrame([{c: np.nan for c in cols}])

    df = wx.copy()

    # Time → seconds for coverage/span
    if "Time" in df.columns:
        t = df["Time"].apply(td_to_seconds)
        span = float(t.max() - t.min()) if t.notna().any() else np.nan
    else:
        span = np.nan

    # Rainfall share
    rain = to_bool_series(df.get("Rainfall", pd.Series(False, index=df.index)))
    rain_rate = float(rain.mean()) if len(rain) else np.nan
    rain_any = bool(rain.any()) if len(rain) else False

    out: Dict[str, float] = {"wx_samples": int(len(df)), "wx_span_s": span, "rain_rate": rain_rate, "rain_any": rain_any}

    # Scalar numeric blocks
    mapping = {
        "AirTemp": "air_temp",
        "TrackTemp": "track_temp",
        "Humidity": "humidity",
        "Pressure": "pressure",
        "WindSpeed": "wind_speed",
    }
    for col, name in mapping.items():
        if col in df.columns:
            out.update(_agg_numeric(df[col], name))
        else:
            out.update({f"{name}_mean": np.nan, f"{name}_std": 0.0, f"{name}_min": np.nan, f"{name}_max": np.nan})

    # Wind direction (circular)
    if "WindDirection" in df.columns:
        out.update(_wind_direction_features(df["WindDirection"]))
    else:
        out.update({"wind_dir_x_mean": np.nan, "wind_dir_y_mean": np.nan, "wind_dir_deg_mean": np.nan})

    return pd.DataFrame([out])
