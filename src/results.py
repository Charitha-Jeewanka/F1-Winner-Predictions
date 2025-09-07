from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.common import td_to_seconds


@dataclass
class ResultsBundle:
    df: pd.DataFrame
    map_number_to_abbr: pd.DataFrame  


def parse_results(results_csv: str) -> ResultsBundle:
    res = pd.read_csv(results_csv)

    if "ClassifiedPosition" in res.columns:
        res = res.rename(columns={"ClassifiedPosition": "finish_pos"})
    elif "Position" in res.columns:
        res = res.rename(columns={"Position": "finish_pos"})
    if "GridPosition" in res.columns:
        res = res.rename(columns={"GridPosition": "grid_pos"})

    if "RaceTime" in res.columns:
        res["race_gap_s"] = res["RaceTime"].apply(td_to_seconds).astype(float)
        res["race_gap_s"] = res["race_gap_s"] - res["race_gap_s"].min(skipna=True)
    else:
        res["race_gap_s"] = np.nan

    res = res.rename(columns={
        "Abbreviation": "Abbreviation",
        "DriverNumber": "driver_number",
        "TeamId": "team",
        "Status": "status",
        "Points": "points",
        "Laps": "laps_completed",
    })

    for c in ["Abbreviation","driver_number","team","grid_pos","finish_pos","status","points","laps_completed","race_gap_s"]:
        if c not in res:
            res[c] = np.nan

    res["grid_pos_num"]   = pd.to_numeric(res["grid_pos"], errors="coerce")
    res["finish_pos_num"] = pd.to_numeric(res["finish_pos"], errors="coerce")
    res["points"]         = pd.to_numeric(res["points"], errors="coerce")
    res["laps_completed"] = pd.to_numeric(res["laps_completed"], errors="coerce")
    res["pos_gain"]       = res["grid_pos_num"] - res["finish_pos_num"]

    map_df = res[["driver_number", "Abbreviation"]].dropna().drop_duplicates().copy()
    map_df["driver_number"] = map_df["driver_number"].astype(str)

    res_std = res[[
        "Abbreviation",
        "driver_number",
        "team",
        "grid_pos_num",
        "finish_pos_num",
        "pos_gain",
        "status",
        "points",
        "laps_completed",
        "race_gap_s",
    ]].rename(columns={"grid_pos_num": "grid_pos", "finish_pos_num": "finish_pos"})

    return ResultsBundle(df=res_std, map_number_to_abbr=map_df)


def harmonize_abbreviation(laps_df: pd.DataFrame, results: ResultsBundle) -> pd.DataFrame:
    if laps_df is None or laps_df.empty:
        return laps_df

    df = laps_df.copy()
    cols_lc = {c.lower(): c for c in df.columns}

    if "abbreviation" in cols_lc:
        ab_col = cols_lc["abbreviation"]
        if ab_col != "Abbreviation":
            df = df.rename(columns={ab_col: "Abbreviation"})
        return df

    if "driver" in cols_lc:
        df["Abbreviation"] = df[cols_lc["driver"]]
        return df

    if "drivernumber" in cols_lc and results is not None and not results.df.empty:
        dn_col = cols_lc["drivernumber"]
        df[dn_col] = df[dn_col].astype(str)
        df = df.merge(
            results.map_number_to_abbr,
            left_on=dn_col,
            right_on="driver_number",
            how="left",
        ).drop(columns=["driver_number"])
        return df

    raise KeyError("Unable to harmonize driver identity: need Abbreviation/Driver/DriverNumber in laps.csv")
