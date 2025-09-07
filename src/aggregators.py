from __future__ import annotations

import abc
import numpy as np
import pandas as pd

from utils.common import (
    convert_time_cols,
    valid_quali_mask,
    valid_race_mask_iqr,
    pace_summaries,
    to_bool_series,
)


class SessionAggregator(abc.ABC):
    @abc.abstractmethod
    def aggregate(self, laps_df: pd.DataFrame | None) -> pd.DataFrame:
        ...


class QualifyingAggregator(SessionAggregator):
    def aggregate(self, laps_df: pd.DataFrame | None) -> pd.DataFrame:
        cols = [
            "Abbreviation",
            "q_laps",
            "q_best_lap_s",
            "q_top3_mean_s",
            "q_median_pace_s",
            "q_pace_std_s",
            "q_p95_s",
            "q_pb_rate",
            "q_max_speed_st",
            "q_max_speed_fl",
            "q_max_speed_i1",
            "q_max_speed_i2",
            "q_compounds_used",
            "q_best_compound",
        ]
        if laps_df is None or laps_df.empty:
            return pd.DataFrame(columns=cols)

        df = laps_df.copy()
        convert_time_cols(df, ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"])
        if "IsPersonalBest" in df.columns:
            df["IsPersonalBest"] = to_bool_series(df["IsPersonalBest"])

        lv = df[valid_quali_mask(df)].copy()
        if "Abbreviation" not in lv.columns or lv["Abbreviation"].isna().all():
            return pd.DataFrame(columns=cols)

        lv["is_best_of_driver"] = lv.groupby("Abbreviation")["LapTime"].transform(lambda s: s == s.min())

        agg_basic = lv.groupby("Abbreviation").agg(
            q_laps=("LapTime", "count"),
            q_pb_rate=("IsPersonalBest", "mean") if "IsPersonalBest" in lv.columns else ("LapTime", lambda _: np.nan),
            q_max_speed_st=("SpeedST", "max") if "SpeedST" in lv.columns else ("LapTime", lambda _: np.nan),
            q_max_speed_fl=("SpeedFL", "max") if "SpeedFL" in lv.columns else ("LapTime", lambda _: np.nan),
            q_max_speed_i1=("SpeedI1", "max") if "SpeedI1" in lv.columns else ("LapTime", lambda _: np.nan),
            q_max_speed_i2=("SpeedI2", "max") if "SpeedI2" in lv.columns else ("LapTime", lambda _: np.nan),
            q_compounds_used=("Compound", "nunique") if "Compound" in lv.columns else ("LapTime", lambda _: 0),
        ).reset_index()

        pace = (
            lv.groupby("Abbreviation")["LapTime"]
            .apply(lambda s: pd.Series(pace_summaries(s)))
            .reset_index()
            .rename(columns={
                "best": "q_best_lap_s",
                "median": "q_median_pace_s",
                "std": "q_pace_std_s",
                "p95": "q_p95_s",
                "top3": "q_top3_mean_s",
            })
        )

        if "Compound" in lv.columns:
            best_comp = (
                lv[lv["is_best_of_driver"]]
                .sort_values(["Abbreviation", "LapTime"])
                .groupby("Abbreviation")["Compound"]
                .first()
                .reset_index()
                .rename(columns={"Compound": "q_best_compound"})
            )
        else:
            best_comp = pd.DataFrame({"Abbreviation": agg_basic["Abbreviation"], "q_best_compound": np.nan})

        return agg_basic.merge(pace, on="Abbreviation", how="outer").merge(best_comp, on="Abbreviation", how="left")


class RaceAggregator(SessionAggregator):
    def aggregate(self, laps_df: pd.DataFrame | None) -> pd.DataFrame:
        cols = [
            "Abbreviation",
            "race_laps",
            "race_best_lap_s",
            "race_median_pace_s",
            "race_pace_std_s",
            "race_p95_s",
            "race_top3_mean_s",
            "stints",
            "pit_stop_count",
            "first_compound",
            "last_compound",
            "laps_SOFT",
            "laps_MEDIUM",
            "laps_HARD",
            "laps_INTERMEDIATE",
            "laps_WET",
            "avg_sector1_s",
            "avg_sector2_s",
            "avg_sector3_s",
            "race_max_speed_st",
            "race_max_speed_fl",
            "race_max_speed_i1",
            "race_max_speed_i2",
        ]
        if laps_df is None or laps_df.empty:
            return pd.DataFrame(columns=cols)

        df = laps_df.copy()
        convert_time_cols(df, ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"])
        if "LapNumber" in df.columns:
            df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")

        lv = df[valid_race_mask_iqr(df)].copy()
        if "Abbreviation" not in lv.columns or lv["Abbreviation"].isna().all():
            return pd.DataFrame(columns=cols)

        pace = (
            lv.groupby("Abbreviation")["LapTime"]
            .apply(lambda s: pd.Series(pace_summaries(s)))
            .reset_index()
            .rename(columns={
                "best": "race_best_lap_s",
                "median": "race_median_pace_s",
                "std": "race_pace_std_s",
                "p95": "race_p95_s",
                "top3": "race_top3_mean_s",
            })
        )

        if "Stint" in lv.columns:
            stints = lv.groupby("Abbreviation")["Stint"].nunique().rename("stints").reset_index()
        else:
            stints = pd.DataFrame({"Abbreviation": lv["Abbreviation"].unique(), "stints": 1})
        stints["pit_stop_count"] = stints["stints"].clip(lower=1) - 1

        if "Compound" in lv.columns:
            ordered = lv.sort_values(["Abbreviation", "LapNumber"])
            first_comp = ordered.groupby("Abbreviation")["Compound"].first().rename("first_compound")
            last_comp  = ordered.groupby("Abbreviation")["Compound"].last().rename("last_compound")
            comp_meta = pd.concat([first_comp, last_comp], axis=1).reset_index()

            comp_counts = (
                lv.pivot_table(index="Abbreviation", columns="Compound", values="LapTime", aggfunc="count")
                .fillna(0).astype(int).reset_index()
            )
            for want in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
                if want not in comp_counts.columns:
                    comp_counts[want] = 0
            comp_counts = comp_counts.rename(columns={
                "SOFT": "laps_SOFT",
                "MEDIUM": "laps_MEDIUM",
                "HARD": "laps_HARD",
                "INTERMEDIATE": "laps_INTERMEDIATE",
                "WET": "laps_WET",
            })
            comp_counts = comp_counts[["Abbreviation", "laps_SOFT", "laps_MEDIUM", "laps_HARD", "laps_INTERMEDIATE", "laps_WET"]]
        else:
            comp_meta = pd.DataFrame({"Abbreviation": lv["Abbreviation"].unique(), "first_compound": np.nan, "last_compound": np.nan})
            comp_counts = pd.DataFrame({
                "Abbreviation": lv["Abbreviation"].unique(),
                "laps_SOFT": 0, "laps_MEDIUM": 0, "laps_HARD": 0, "laps_INTERMEDIATE": 0, "laps_WET": 0,
            })

        sec_agg = lv.groupby("Abbreviation").agg(
            avg_sector1_s=("Sector1Time", "mean") if "Sector1Time" in lv.columns else ("LapTime", lambda _: np.nan),
            avg_sector2_s=("Sector2Time", "mean") if "Sector2Time" in lv.columns else ("LapTime", lambda _: np.nan),
            avg_sector3_s=("Sector3Time", "mean") if "Sector3Time" in lv.columns else ("LapTime", lambda _: np.nan),
            race_laps=("LapNumber", "count") if "LapNumber" in lv.columns else ("LapTime", "count"),
            race_max_speed_st=("SpeedST", "max") if "SpeedST" in lv.columns else ("LapTime", lambda _: np.nan),
            race_max_speed_fl=("SpeedFL", "max") if "SpeedFL" in lv.columns else ("LapTime", lambda _: np.nan),
            race_max_speed_i1=("SpeedI1", "max") if "SpeedI1" in lv.columns else ("LapTime", lambda _: np.nan),
            race_max_speed_i2=("SpeedI2", "max") if "SpeedI2" in lv.columns else ("LapTime", lambda _: np.nan),
        ).reset_index()

        out = (
            sec_agg.merge(pace, on="Abbreviation", how="left")
            .merge(stints, on="Abbreviation", how="left")
            .merge(comp_meta, on="Abbreviation", how="left")
            .merge(comp_counts, on="Abbreviation", how="left")
        )
        return out
