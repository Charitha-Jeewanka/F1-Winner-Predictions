import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.metrics.race_metrics import compute_classification_metrics, compute_race_topk_metrics
from utils.mlflow_logger import parent_run, child_run, log_params, log_metrics, log_artifacts, log_dict

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names"
)


def _load_test_split(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    X_test = pd.DataFrame(data["X_test"])
    y_test = pd.Series(data["y_test"])
    g_test = pd.Series(data["g_test"])
    num_cols = data.get("num_cols", [])
    cat_cols = data.get("cat_cols", [])
    return X_test, y_test, g_test, num_cols, cat_cols



def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: Path):
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _save_json(obj: dict, path: Path):
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _plot_roc(y_true, y_proba, out_path, name: str):
    disp = RocCurveDisplay.from_predictions(y_true, y_proba, name=name)
    fig = disp.figure_
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def _plot_cm(y_true, y_pred, out_path, name: str):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, colorbar=False)
    fig = disp.figure_
    fig.suptitle(f"Confusion Matrix — {name}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(cfg_path: str):

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models_dir = Path(cfg["data_paths"]["models_dir"]).resolve()
    reports_dir = Path(cfg["data_paths"]["reports_dir"]).resolve()
    artifacts_dir = Path(cfg["data_paths"]["artifacts_dir"]).resolve()
    mlf_cfg = cfg.get("mlflow", {})

    X_test, y_test, g_test, _, _ = _load_test_split(artifacts_dir / "test_split.json")
    threshold = cfg["metrics"].get("threshold", 0.5)
    ks = cfg["metrics"].get("ks", [1, 3])

    rows = []

    with parent_run(mlf_cfg, phase="evaluation", run_name="evaluation"):
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir(): 
                continue
            model_name = model_dir.name
            model_path = model_dir / "model.joblib"
            if not model_path.exists():
                continue

            print(f"Evaluating {model_name} ...")

            pipe = joblib.load(model_path)

            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                y_proba = pipe.predict_proba(X_test)[:, 1]
            else:
                decision = pipe.decision_function(X_test)
                y_proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
            y_pred = (y_proba >= threshold).astype(int)

            cls_metrics = compute_classification_metrics(y_test, y_pred, y_proba)
            race_metrics = compute_race_topk_metrics(y_true=y_test, y_proba=y_proba, groups=g_test, ks=ks)
            metrics = {"model": model_name, **cls_metrics, **race_metrics}

            print(
            "  test_roc_auc={:.3f}  test_pr_auc={:.3f}  race_top1={:.3f}  race_top3={:.3f}".format(
                cls_metrics["roc_auc"],
                cls_metrics["pr_auc"],
                race_metrics.get("race_top1", float("nan")),
                race_metrics.get("race_top3", float("nan")),
            )
        )

            
            (model_dir / "eval").mkdir(exist_ok=True, parents=True)
            with open(model_dir / "eval" / "metrics_test.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            
            _plot_roc(y_test, y_proba, out_path=model_dir / "eval" / "roc_test.png", name=f"{model_name}")
            _plot_cm(y_test, y_pred, out_path=model_dir / "eval" / "cm_test.png", name=f"{model_name}")

            
            pd.DataFrame({
                "event": g_test,
                "y_true": y_test,
                "y_proba": y_proba,
                "y_pred": y_pred,
            }).to_csv(model_dir / "eval" / "test_predictions.csv", index=False)

            with child_run(model_name, stage="test"):
                log_metrics({f"test_{k}": v for k, v in cls_metrics.items()})
                log_metrics({f"test_{k}": v for k, v in race_metrics.items()})
                log_artifacts(model_dir / "eval", artifact_path=f"{model_name}/eval")

            rows.append(metrics)

        leaderboard = pd.DataFrame(rows)
        if not leaderboard.empty:
            sort_cols = [c for c in ["race_top1", "roc_auc"] if c in leaderboard.columns]
            leaderboard = leaderboard.sort_values(by=sort_cols, ascending=[False]*len(sort_cols))
            out_csv = reports_dir / "test_leaderboard.csv"
            _ensure_dir(out_csv.parent)
            leaderboard.to_csv(out_csv, index=False)
            log_artifacts(reports_dir, artifact_path="reports")

    print("\nTest leaderboard:", reports_dir / "test_leaderboard.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)