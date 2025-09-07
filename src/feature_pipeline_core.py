from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import yaml

from utils.common import normalize_paths
from src.results import parse_results, harmonize_abbreviation
from src.aggregators import SessionAggregator, QualifyingAggregator, RaceAggregator
from src.weather import aggregate_weather_session  


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "_manifests" / "global_manifest.csv"
OUT_DIR = ROOT / "data" / "processed"
SKIP_EXISTING = False


class FeatureEngineeringPipeline:
    def __init__(self) -> None:
        self.manifest_csv = self._load_manifest_path()
        self.quali_agg: SessionAggregator = QualifyingAggregator()
        self.race_agg: SessionAggregator = RaceAggregator()

    def _load_manifest_path(self) -> Path:
        cfg_path = ROOT / "config.yaml"
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text())
                m = cfg.get("data_paths", {}).get("manifest")
                if m:
                    p = Path(m)
                    return p if p.is_absolute() else (ROOT / p)
            except Exception as e:
                logging.warning("Failed to read config.yaml (%s); falling back to default manifest.", e)
        return DEFAULT_MANIFEST

    def _collect_events(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logging.info("Reading manifest: %s", self.manifest_csv)
        man = pd.read_csv(self.manifest_csv)
        man = normalize_paths(man)
        race_res = man[(man["session"] == "R") & (man["table"] == "results")]
        events = race_res[["year", "round"]].drop_duplicates().sort_values(["year", "round"])
        logging.info("Discovered %d events with race results.", len(events))
        return events, man

    def _paths_for_event(
        self, man: pd.DataFrame, year: int, rnd: int
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
        def pick(sess: str, table: str) -> Optional[str]:
            rows = man[(man["year"] == year) & (man["round"] == rnd) & (man["session"] == sess) & (man["table"] == table)]
            return rows["path"].iloc[0] if len(rows) else None

        # return Q laps, R laps, R results, Q weather, R weather
        return pick("Q", "laps"), pick("R", "laps"), pick("R", "results"), pick("Q", "weather"), pick("R", "weather")

    def build_event(self, man: pd.DataFrame, year: int, rnd: int) -> Optional[pd.DataFrame]:
        q_laps_csv, r_laps_csv, r_results_csv, q_wx_csv, r_wx_csv = self._paths_for_event(man, year, rnd)
        if r_results_csv is None:
            logging.warning("Skipping %s_%s: manifest missing race results path.", year, rnd)
            return None

        def _abs(p: Optional[str]) -> Optional[Path]:
            if not p:
                return None
            path = Path(p)
            return path if path.is_absolute() else (ROOT / path)

        r_results_path = _abs(r_results_csv)
        if not r_results_path or not r_results_path.exists():
            logging.warning("Skipping %s_%s: race results not found at %s", year, rnd, r_results_path)
            return None

        results = parse_results(str(r_results_path))

        q_laps_path = _abs(q_laps_csv)
        r_laps_path = _abs(r_laps_csv)
        laps_q = pd.read_csv(q_laps_path) if q_laps_path and q_laps_path.exists() else None
        laps_r = pd.read_csv(r_laps_path) if r_laps_path and r_laps_path.exists() else None
        if laps_q is not None:
            laps_q = harmonize_abbreviation(laps_q, results)
        if laps_r is not None:
            laps_r = harmonize_abbreviation(laps_r, results)

        # WEATHER (session-level summaries)
        q_wx_path = _abs(q_wx_csv)
        r_wx_path = _abs(r_wx_csv)
        wx_q = aggregate_weather_session(pd.read_csv(q_wx_path)) if q_wx_path and q_wx_path.exists() else None
        wx_r = aggregate_weather_session(pd.read_csv(r_wx_path)) if r_wx_path and r_wx_path.exists() else None

        if wx_q is not None and not wx_q.empty:
            wx_q = wx_q.add_prefix("wx_q_")
        if wx_r is not None and not wx_r.empty:
            wx_r = wx_r.add_prefix("wx_r_")

        # AGGREGATIONS
        quali = self.quali_agg.aggregate(laps_q)
        race = self.race_agg.aggregate(laps_r)

        merged = results.df.merge(race, on="Abbreviation", how="left").merge(quali, on="Abbreviation", how="left")
        merged.insert(0, "event_code", f"{year}_{rnd}")
        merged.insert(1, "year", year)
        merged.insert(2, "round", rnd)

        # Broadcast weather features to all driver rows in this event
        if wx_r is not None and not wx_r.empty:
            for k, v in wx_r.iloc[0].items():
                merged[k] = v
        if wx_q is not None and not wx_q.empty:
            for k, v in wx_q.iloc[0].items():
                merged[k] = v

        if merged["finish_pos"].notna().any():
            merged = merged.sort_values(["finish_pos", "race_gap_s"], na_position="last")

        return merged

    def run_all(self) -> None:
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        events, man = self._collect_events()
        logging.info("Processing %d events ...", len(events))

        all_chunks: List[pd.DataFrame] = []
        for _, row in events.iterrows():
            year, rnd = int(row["year"]), int(row["round"])
            out_path = OUT_DIR / f"{year}_{rnd}_race_dataset.csv"

            if SKIP_EXISTING and out_path.exists():
                logging.info("Exists, skipping: %s", out_path)
                try:
                    all_chunks.append(pd.read_csv(out_path))
                except Exception:
                    logging.info("Rebuilding unreadable file: %s", out_path)
                    df = self.build_event(man, year, rnd)
                    if df is not None:
                        df.to_csv(out_path, index=False)
                        all_chunks.append(df)
                continue

            logging.info("Building %d_%d ...", year, rnd)
            df = self.build_event(man, year, rnd)
            if df is None:
                continue
            df.to_csv(out_path, index=False)
            logging.info("Wrote %s  rows=%d  cols=%d", out_path, len(df), len(df.columns))
            all_chunks.append(df)

        if all_chunks:
            big = pd.concat(all_chunks, ignore_index=True)
            big_out = OUT_DIR / "race_dataset_all.csv"
            big.to_csv(big_out, index=False)
            logging.info("Combined dataset: %s  rows=%d  cols=%d", big_out, len(big), len(big.columns))
        else:
            logging.warning("No events processed.")
