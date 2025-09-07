from __future__ import annotations
import argparse, logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

log = logging.getLogger("handle_missing_values")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TIME_COLS: Dict[str, List[str]] = {
    "results": ["Q1","Q2","Q3","RaceTime"],
    "laps": ["Time","LapTime","Sector1Time","Sector2Time","Sector3Time","LapStartTime"],
    "weather": [],
}

def _processed_path_from_manifest(raw_path: str, processed_root: Path) -> Path:
    p = Path(raw_path.replace("\\", "/"))
    parts = list(p.parts)
    try:
        i = parts.index("raw")
        parts[i] = "processed"
        return processed_root.parent.joinpath(*parts[1:]) if processed_root.name == "processed" else processed_root.joinpath(*parts[i+1:])
    except ValueError:
        return processed_root / p.name

def _read_processed(row: dict, processed_root: Path) -> Tuple[pd.DataFrame, Path]:
    proc_path = _processed_path_from_manifest(str(row["path"]), processed_root)
    if not proc_path.exists():
        raise FileNotFoundError(f"Processed file not found: {proc_path} (did you run feature_selection first?)")
    return pd.read_csv(proc_path), proc_path

def _to_td(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.timedelta64):
        return s
    return pd.to_timedelta(s, errors="coerce")

def _td_to_seconds(s: pd.Series) -> pd.Series:
    return _to_td(s).dt.total_seconds()

def _seconds_to_td(s: pd.Series) -> pd.Series:
    return pd.to_timedelta(s, unit="s", errors="coerce")

def _numeric_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    ex = set(exclude)
    cols: List[str] = []
    for c in df.columns:
        if c in ex: 
            continue
        dt = df[c].dtype
        if np.issubdtype(dt, np.number):
            if dt == bool or dt == np.bool_:
                continue
            cols.append(c)
    return cols

def _impute_group_median(df: pd.DataFrame, key: str, cols: List[str]) -> pd.DataFrame:
    if not cols or df.empty:
        return df
    out = df.copy()
    g = out.groupby(key, dropna=False)
    for c in cols:
        med = g[c].transform("median")
        gmed = out[c].median()
        out[c] = out[c].where(out[c].notna(), med.fillna(gmed))
    return out

def _process_results(res: pd.DataFrame) -> pd.DataFrame:
    if "Abbreviation" not in res.columns:
        raise ValueError("results requires 'Abbreviation' for per-driver median imputation.")
    work = res.copy()
    tcols = [c for c in TIME_COLS["results"] if c in work.columns]
    shadows = []
    for c in tcols:
        sc = f"__sec__{c}"
        work[sc] = _td_to_seconds(work[c])
        shadows.append(sc)
    num_cols = _numeric_cols(work, exclude=["Abbreviation"] + tcols + shadows)
    work = _impute_group_median(work, "Abbreviation", shadows + num_cols)
    for c in tcols:
        sc = f"__sec__{c}"; work[c] = _seconds_to_td(work[sc]); work.drop(columns=[sc], inplace=True, errors="ignore")
    return work

def _build_lookup(results_imp: pd.DataFrame) -> pd.DataFrame:
    need = {"DriverNumber","Abbreviation"}
    if not need.issubset(results_imp.columns):
        raise ValueError("Driver lookup needs DriverNumber and Abbreviation in results.")
    return results_imp.loc[:, ["DriverNumber","Abbreviation"]].dropna().drop_duplicates()

def _process_laps(laps: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    work = laps.copy()
    if "Abbreviation" not in work.columns:
        before = len(work)
        work = work.merge(lookup, on="DriverNumber", how="left")
        miss = work["Abbreviation"].isna().sum()
        if miss:
            log.warning(f"[laps] Abbreviation missing for {miss}/{before} rows after lookup; "
                        "those rows will use global medians.")
            work["Abbreviation"] = work["Abbreviation"].fillna("__GLOBAL__")
    tcols = [c for c in TIME_COLS["laps"] if c in work.columns]
    shadows = []
    for c in tcols:
        sc = f"__sec__{c}"
        work[sc] = _td_to_seconds(work[c])
        shadows.append(sc)
    num_cols = _numeric_cols(work, exclude=["Abbreviation"] + tcols + shadows)
    work = _impute_group_median(work, "Abbreviation", shadows + num_cols)
    for c in tcols:
        sc = f"__sec__{c}"; work[c] = _seconds_to_td(work[sc]); work.drop(columns=[sc], inplace=True, errors="ignore")
    work["Abbreviation"] = work["Abbreviation"].replace({"__GLOBAL__": np.nan})
    return work

def run(manifest_path: Path, processed_root: Path) -> None:
    mf = pd.read_csv(manifest_path)
    need = {"year","round","session","table","rows","path"}
    if not need.issubset(mf.columns):
        raise ValueError(f"Manifest missing: {need - set(mf.columns)}")

    total = 0
    for (year, rnd, sess), grp in mf.groupby(["year","round","session"], dropna=False):
        lookup = None
        rows_results = grp[grp["table"].str.lower() == "results"]
        if len(rows_results):
            res_df, res_path = _read_processed(rows_results.iloc[0].to_dict(), processed_root)
            res_imp = _process_results(res_df)
            res_imp.to_csv(res_path, index=False)
            lookup = _build_lookup(res_imp)
            total += len(res_imp)
            log.info(f"[{year}-{rnd}-{sess}] results imputed -> {res_path}")

        rows_laps = grp[grp["table"].str.lower() == "laps"]
        if len(rows_laps):
            if lookup is None:
                log.warning(f"[{year}-{rnd}-{sess}] laps present without results; falling back to global medians.")
                lookup = pd.DataFrame(columns=["DriverNumber","Abbreviation"])
            laps_df, laps_path = _read_processed(rows_laps.iloc[0].to_dict(), processed_root)
            laps_imp = _process_laps(laps_df, lookup)
            laps_imp.to_csv(laps_path, index=False)
            total += len(laps_imp)
            log.info(f"[{year}-{rnd}-{sess}] laps imputed -> {laps_path}")


    log.info(f"Done imputing (processed-only). Total rows written: {total}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Impute missing values per driver (processed-only).")
    ap.add_argument("--manifest", default="data/raw/_manifests/global_manifest.csv", type=Path)
    ap.add_argument("--processed-root", default="data/processed", type=Path)
    args = ap.parse_args()
    run(args.manifest, args.processed_root)
