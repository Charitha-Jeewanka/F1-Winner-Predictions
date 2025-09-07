import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV

from utils.data import (
    load_and_prepare,
    grouped_train_val_test_split,
    build_preprocessor,
    infer_coltypes,
)
from src.models.registry import build_estimator
from src.metrics.race_metrics import compute_classification_metrics, compute_race_topk_metrics
from utils.mlflow_logger import parent_run, child_run, log_params, log_metrics, log_artifacts, log_model_sklearn, log_dict


def _save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(cfg_path: str):

    def _save_json(data: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _save_csv(df: pd.DataFrame, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.RandomState(cfg.get("random_seed", 42))

    data_path = Path(cfg["data_paths"]["dataset"]).resolve()
    models_dir = Path(cfg["data_paths"]["models_dir"]).resolve()
    reports_dir = Path(cfg["data_paths"]["reports_dir"]).resolve()
    artifacts_dir = Path(cfg["data_paths"]["artifacts_dir"]).resolve()
    for p in (models_dir, reports_dir, artifacts_dir):
        p.mkdir(parents=True, exist_ok=True)

    from utils.data import (
        load_and_prepare, grouped_train_val_test_split,
        build_preprocessor, infer_coltypes
    )
    from src.metrics.race_metrics import (
        compute_classification_metrics, compute_race_topk_metrics
    )
    from src.models.registry import build_estimator

    df, target_col, group_col = load_and_prepare(
        data_path,
        target_col=cfg["columns"]["target"],
        finish_pos_col=cfg["columns"].get("finish_pos", "finish_pos"),
        group_col=cfg["columns"]["group"],
        leakage_guard=cfg["columns"].get("leakage_guard", []),
        drop_columns=cfg["columns"].get("drop_columns", []),
    )

    cat_hints = set(cfg["columns"].get("categorical_hints", []))
    feat_df = df.drop(columns=[target_col, group_col])
    num_cols, cat_cols = infer_coltypes(feat_df, cat_hint_names=cat_hints)

    (X_train, y_train, g_train), (X_val, y_val, g_val), (X_test, y_test, g_test) = grouped_train_val_test_split(
        df=df,
        group_col=group_col,
        target_col=target_col,
        test_size=cfg["split"].get("test_size", 0.15),
        val_size=cfg["split"].get("val_size", 0.15),
        rng=rng,
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("Train split has no positive or no negative samples. Adjust split or add data.")
    pos_weight = float(neg) / float(pos)

    enabled_models = cfg["models"]["enabled"]
    leaderboard_rows = []

    mlf_cfg = cfg.get("mlflow", {})

    with parent_run(mlf_cfg, phase="training", run_name="training", tags={"dataset": data_path.name}):
        for model_name in enabled_models:
            params = (cfg["models"].get(model_name, {}) or {}).get("params", {})
            print(f"\n=== Training {model_name} ===")

            preproc = build_preprocessor(model_name, num_cols, cat_cols)
            estimator = build_estimator(
                model_name,
                params=params,
                pos_weight=pos_weight,
                random_state=int(rng.randint(0, 1_000_000)),
            )

            pipe = Pipeline(steps=[
                ("preprocess", preproc),
                ("variance", VarianceThreshold(threshold=0.0)),
                ("clf", estimator),
            ])

            
            with child_run(model_name, stage="train"):
                log_params({**params, "pos_weight": pos_weight})

                pipe.fit(X_train, y_train)

                cal_cfg = cfg.get("calibration", {})
                if cal_cfg.get("enabled", False) and model_name in (cal_cfg.get("models") or []):
                    method = cal_cfg.get("method", "sigmoid")

                    if len(np.unique(y_val)) >= 2:
                        Xt_val = pipe.named_steps["variance"].transform(
                            pipe.named_steps["preprocess"].transform(X_val)
                        )

                        base = pipe.named_steps["clf"]
                        try:
                            cal = CalibratedClassifierCV(estimator=base, method=method, cv="prefit")
                        except TypeError:
                            cal = CalibratedClassifierCV(base_estimator=base, method=method, cv="prefit")

                        try:
                            cal.fit(Xt_val, y_val)
                            
                            pipe.steps[-1] = ("clf", cal)
                        except Exception as e:
                            print(f"[calibration] Skipping calibration for {model_name}: {e}")
                    else:
                        print(f"[calibration] Skipping calibration for {model_name}: only one class in validation set")

                
                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    y_val_proba = pipe.predict_proba(X_val)[:, 1]
                else:
                    decision = pipe.decision_function(X_val)
                    y_val_proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
                thr = cfg["metrics"].get("threshold", 0.5)
                y_val_pred = (y_val_proba >= thr).astype(int)

                cls_metrics = compute_classification_metrics(y_val, y_val_pred, y_val_proba)
                race_metrics = compute_race_topk_metrics(
                    y_true=y_val, y_proba=y_val_proba, groups=g_val, ks=cfg["metrics"].get("ks", [1, 3])
                )

                metrics = {
                    "model": model_name,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "pos_weight": pos_weight,
                    "validation": {**cls_metrics, **race_metrics},
                    "params": params,
                }

                
                model_dir = models_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipe, model_dir / "model.joblib")
                _save_json(metrics, model_dir / "metrics_val.json")

                val_pred_df = pd.DataFrame({
                    "event": g_val,
                    "y_true": y_val,
                    "y_proba": y_val_proba,
                    "y_pred": y_val_pred,
                }).reset_index(drop=True)
                _save_csv(val_pred_df, model_dir / "val_predictions.csv")

                
                log_metrics({f"val_{k}": v for k, v in cls_metrics.items()})
                log_metrics({f"val_{k}": v for k, v in race_metrics.items()})
                log_artifacts(model_dir, artifact_path=f"{model_name}")
                log_model_sklearn(pipe, subname="sklearn_model")
                log_dict(metrics, out_name=f"{model_name}/metrics_val.json")

               
                leaderboard_rows.append({
                    "model": model_name,
                    **{k: v for k, v in cls_metrics.items()},
                    **{k: v for k, v in race_metrics.items()},
                })

       
        leaderboard = pd.DataFrame(leaderboard_rows)
        sort_cols = [c for c in ["race_top1", "roc_auc"] if c in leaderboard.columns]
        leaderboard = leaderboard.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
        _save_csv(leaderboard, reports_dir / "validation_leaderboard.csv")

        
        split_dump = {
            "X_test": X_test.reset_index(drop=True).to_dict(orient="list"),
            "y_test": y_test.reset_index(drop=True).tolist(),
            "g_test": g_test.reset_index(drop=True).tolist(),
            "num_cols": list(num_cols),
            "cat_cols": list(cat_cols),
        }
        _save_json(split_dump, artifacts_dir / "test_split.json")

        log_artifacts(reports_dir, artifact_path="reports")

    print("\nDone. Models saved in:", models_dir)
    print("Validation leaderboard:", reports_dir / "validation_leaderboard.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)