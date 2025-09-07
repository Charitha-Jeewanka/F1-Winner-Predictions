from __future__ import annotations
import argparse, logging
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd

log = logging.getLogger("feature_selection")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RESULTS_KEEP = [
    "Abbreviation","DriverNumber","TeamId",
    "Position","ClassifiedPosition","GridPosition",
    "Q1","Q2","Q3","RaceTime","Status","Points","Laps",
]

LAPS_KEEP = [
    "Time","DriverNumber","LapNumber","LapTime","Stint",
    "Sector1Time","Sector2Time","Sector3Time",
    "SpeedI1","SpeedI2","SpeedFL","SpeedST",
    "IsPersonalBest","Compound","TyreLife","FreshTyre",
    "LapStartTime","TrackStatus","Position","Deleted",
]

KEEP_MAP: Dict[str, Union[str, List[str]]] = {
    "results": RESULTS_KEEP,
    "laps": LAPS_KEEP,
    "weather": "ALL",
}

def _processed_path_from_manifest(raw_path: str, out_root: Path) -> Path:
    p = Path(raw_path.replace("\\", "/"))
    parts = list(p.parts)
    try:
        i = parts.index("raw")
        parts[i] = "processed"
        return out_root.parent.joinpath(*parts[1:]) if out_root.name == "processed" else out_root.joinpath(*parts[i+1:])
    except ValueError:
        return out_root / p.name

def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _select_cols(df: pd.DataFrame, table: str) -> pd.DataFrame:
    tbl = table.lower()
    keep = KEEP_MAP.get(tbl)
    if keep is None:
        raise ValueError(f"Unknown table '{table}'")

    if tbl == "results" and "RaceTime" not in df.columns and "Time" in df.columns:
        df = df.rename(columns={"Time": "RaceTime"})

    if keep == "ALL":
        return df.copy()

    existing = [c for c in keep if c in df.columns]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        log.warning(f"[{table}] missing columns (skipped): {missing}")
    return df[existing].copy()

def run(manifest_path: Path, out_root: Path) -> None:
    mf = _load_csv(manifest_path)
    need = {"year","round","session","table","rows","path"}
    if not need.issubset(mf.columns): raise ValueError(f"Manifest missing: {need - set(mf.columns)}")

    total = 0
    for rec in mf.to_dict("records"):
        table = str(rec["table"]).lower()
        if table not in KEEP_MAP: 
            continue
        src = Path(rec["path"])
        df = _load_csv(src)
        sel = _select_cols(df, table)
        dst = _processed_path_from_manifest(str(src), Path("data/processed"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        sel.to_csv(dst, index=False)
        log.info(f"[{table}] wrote {len(sel):>6} rows -> {dst}")
        total += len(sel)
    log.info(f"Done. Total processed rows: {total}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/raw/_manifests/global_manifest.csv", type=Path)
    ap.add_argument("--out-root", default="data/processed", type=Path)
    args = ap.parse_args()
    run(args.manifest, args.out_root)
