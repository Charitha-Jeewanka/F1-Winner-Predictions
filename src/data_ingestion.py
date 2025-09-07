from __future__ import annotations
import re, time, logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import fastf1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        pass


def _parse_session_spec(spec: str) -> Tuple[int, int, str]:
    s = spec.strip()
    m = re.match(r"^\s*(\d{4})\s*,\s*(\d{1,2})\s*,\s*([A-Za-z0-9]+)\s*$", s)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3).upper()
    kv = {}
    if "&" in s or "=" in s:
        for part in s.replace(" ", "").split("&"):
            if "=" in part:
                k, v = part.split("=", 1); kv[k.lower()] = v
    else:
        for part in re.split(r"\s+", s):
            if ":" in part:
                k, v = part.split(":", 1); kv[k.lower()] = v
    return int(kv["year"]), int(kv["round"]), kv.get("session", "R").upper()


@dataclass
class FastF1SessionIngestor(DataIngestor):
    cache_dir: Path
    save_root: Path
    retries: int = 3
    retry_sleep_sec: float = 2.0
    load_telemetry: bool = False

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.save_root = Path(self.save_root); self.save_root.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))

    def _load_session(self, year: int, rnd: int, sess: str):
        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                log.info(f"Loading session {year}-{rnd}-{sess} (attempt {attempt})")
                s = fastf1.get_session(year, rnd, sess)
                s.load(weather=True, telemetry=self.load_telemetry, laps=True)
                return s
            except Exception as e:
                last_err = e
                log.warning(f"Retry {attempt} failed for {year}-{rnd}-{sess}: {e}")
                time.sleep(self.retry_sleep_sec)
        raise RuntimeError(f"Session load failed for {year}-{rnd}-{sess}") from last_err

    def _save_df(self, df: pd.DataFrame, out_dir: Path, name: str) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{name}.csv"
        df.reset_index(drop=True).to_csv(p, index=False)
        return p

    def ingest(self, file_path_or_link: str) -> pd.DataFrame:
        year, rnd, sess = _parse_session_spec(file_path_or_link)
        out_dir = self.save_root / f"{year}_{rnd}_{sess}"
        try:
            session = self._load_session(year, rnd, sess)
        except Exception as e:
            log.error(f"Skipping {year}-{rnd}-{sess}: {e}")
            return pd.DataFrame([{"year": year, "round": rnd, "session": sess, "table": "ERROR", "rows": 0, "path": str(e)}])

        tables: Dict[str, pd.DataFrame] = {}
        if getattr(session, "results", None) is not None:
            tables["results"] = pd.DataFrame(session.results)
        if getattr(session, "laps", None) is not None:
            tables["laps"] = pd.DataFrame(session.laps)
        if getattr(session, "weather_data", None) is not None:
            tables["weather"] = pd.DataFrame(session.weather_data)

        recs = []
        for name, df in tables.items():
            path = self._save_df(df, out_dir, name)
            recs.append({"year": year, "round": rnd, "session": sess, "table": name, "rows": len(df), "path": str(path)})
            log.info(f"Saved {name} ({len(df)} rows) for {year}-{rnd}-{sess}")

        manifest = pd.DataFrame(recs, columns=["year", "round", "session", "table", "rows", "path"])
        self._save_df(manifest, out_dir, "manifest")
        return manifest


def ingest_seasons(ingestor: FastF1SessionIngestor, years: List[int], sessions: List[str]) -> pd.DataFrame:
    all_manifests = []
    for y in years:
        try:
            sched = fastf1.get_event_schedule(y, include_testing=False)
            rounds = sorted(pd.Series(sched["RoundNumber"]).dropna().astype(int).unique().tolist())
        except Exception as e:
            log.warning(f"Schedule fetch failed for {y}: {e}")
            rounds = list(range(1, 25))
        for r in rounds:
            for sess in sessions:
                spec = f"{y},{r},{sess}"
                log.info(f"Ingesting {spec}")
                mf = ingestor.ingest(spec)
                all_manifests.append(mf)

    out = pd.concat(all_manifests, ignore_index=True) if all_manifests else pd.DataFrame()
    global_manifest_dir = ingestor.save_root / "_manifests"
    global_manifest_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(global_manifest_dir / "manifest_2022_2025_Q_R.csv", index=False)
    return out


if __name__ == "__main__":
    cache = Path("./.fastf1_cache")
    out = Path("./data/raw")
    ing = FastF1SessionIngestor(cache_dir=cache, save_root=out, load_telemetry=False)
    manifest = ingest_seasons(ing, years=list(range(2024, 2026)), sessions=["Q", "R"])
    log.info("Ingestion complete")
    print(manifest.tail(20))
