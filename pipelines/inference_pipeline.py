import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from utils.inference import add_quali_seconds, align_to_training_columns
from utils.mlflow_logger import parent_run, log_params, log_metrics, log_artifacts, log_dict

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _load_train_schema(artifacts_dir: Path):
    with open(artifacts_dir / "test_split.json", "r", encoding="utf-8") as f:
        js = json.load(f)
    num_cols = js.get("num_cols", [])
    cat_cols = js.get("cat_cols", [])
    return num_cols, cat_cols


def _load_models(models_dir: Path, only: list[str] | None = None):
    out = {}
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if only and name not in only:
            continue
        mpath = d / "model.joblib"
        if mpath.exists():
            out[name] = joblib.load(mpath)
    if not out:
        raise RuntimeError("No models found to load.")
    return out


def _load_val_weights(models_dir: Path, model_names: list[str]) -> dict[str, float]:
    weights = {}
    for name in model_names:
        mfile = models_dir / name / "metrics_val.json"
        w = 1.0
        try:
            with open(mfile, "r", encoding="utf-8") as f:
                js = json.load(f)
            w = float(js.get("validation", {}).get("race_top1", 1.0))
        except Exception:
            pass
        weights[name] = w
    s = sum(weights.values())
    if s <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / s for k, v in weights.items()}


def _prepare_quali_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {
        "POS": "grid_pos",
        "NO": "driver_number",
        "TEAM": "team",
        "DRIVER": "driver",
        "LAPS": "quali_laps",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    if "grid_pos" in df.columns:
        df["grid_pos"] = pd.to_numeric(df["grid_pos"], errors="coerce")
    if "driver_number" in df.columns:
        df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce")

    if "Abbreviation" not in df.columns:
        df["Abbreviation"] = "missing"

    df = add_quali_seconds(df)
    return df


def main(cfg_path: str, input_override: str | None = None, event_code_override: str | None = None):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data_paths"]
    infer_cfg = cfg.get("inference", {})
    mlf_cfg = cfg.get("mlflow", {})

    models_dir = Path(data_cfg["models_dir"]).resolve()
    reports_dir = Path(data_cfg["reports_dir"]).resolve()
    artifacts_dir = Path(data_cfg["artifacts_dir"]).resolve()

    input_csv = Path(input_override or infer_cfg.get("input_csv", "data/inference/this_weekend_quali.csv")).resolve()
    event_code = event_code_override or infer_cfg.get("event_code", "this_event")
    top_k = int(infer_cfg.get("top_k", 3))
    ensemble_mode = (infer_cfg.get("ensemble", "weighted") or "weighted").lower()

    only_models = infer_cfg.get("models")
    if only_models:
        only_models = [m.lower() for m in only_models]

    with parent_run(mlf_cfg, phase="inference", run_name=f"inference:{event_code}", tags={"event_code": event_code}):
        
        exp_num, exp_cat = _load_train_schema(artifacts_dir)
        models = _load_models(models_dir, only=only_models)

        
        raw = _prepare_quali_csv(input_csv)

        
        keep_cols = [c for c in ["driver", "team", "grid_pos", "driver_number", "Abbreviation"] if c in raw.columns]

       
        X = align_to_training_columns(raw, expected_num=exp_num, expected_cat=exp_cat)

        
        log_params({
            "ensemble_mode": ensemble_mode,
            "top_k": top_k,
            "n_models": len(models),
            "used_models": ",".join(sorted(models.keys())),
            "input_csv": str(input_csv),
            "event_code": str(event_code),
        })

       
        scores = {}
        for name, pipe in models.items():
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                proba = pipe.predict_proba(X)[:, 1]
            else:
                decision = pipe.decision_function(X)
                proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
            scores[name] = proba

        preds = pd.DataFrame({"event": event_code, **{c: raw[c] for c in keep_cols}})
        for name, proba in scores.items():
            preds[f"proba_{name}"] = proba

        
        if ensemble_mode == "mean":
            preds["proba_ensemble"] = np.vstack(list(scores.values())).mean(axis=0)
            weights = {n: 1.0 / max(len(scores), 1) for n in scores.keys()}
        elif ensemble_mode == "weighted":
            weights = _load_val_weights(models_dir, list(models.keys()))
            stacked = np.vstack([scores[n] * weights.get(n, 0.0) for n in models.keys()])
            preds["proba_ensemble"] = stacked.sum(axis=0)
        else:  
            k_rrf = 60  
            weights = _load_val_weights(models_dir, list(models.keys()))
           
            rank_mat = {name: pd.Series(scores[name]).rank(method="min", ascending=False).to_numpy()
                        for name in models.keys()}
            rrf = None
            for name, r in rank_mat.items():
                contrib = (weights.get(name, 0.0)) / (k_rrf + r)
                rrf = contrib if rrf is None else (rrf + contrib)
            preds["proba_ensemble"] = rrf  

       
        score_cols = [c for c in preds.columns if c.startswith("proba_")]
        for c in score_cols:
            preds[f"rank_{c}"] = preds[c].rank(method="first", ascending=False)

        
        out_dir = reports_dir / "predictions" / str(event_code)
        _ensure_dir(out_dir)
        preds_sorted = preds.sort_values("proba_ensemble", ascending=False)
        preds_sorted.to_csv(out_dir / "predictions.csv", index=False)

       
        top_df = preds_sorted.head(top_k)
        top_df.to_csv(out_dir / f"top{top_k}.csv", index=False)

       
        log_artifacts(out_dir, artifact_path=f"predictions/{event_code}")
        try:
            top_json = json.loads(top_df.to_json(orient="records"))
        except Exception:
            top_json = {}
        log_dict(top_json, out_name=f"predictions/{event_code}/top{top_k}.json")
        log_dict({"ensemble_mode": ensemble_mode, "weights": weights}, out_name=f"predictions/{event_code}/ensemble_weights.json")

    
        print("=== Ensemble Top-{} ===".format(top_k))
        print(top_df[keep_cols + ["proba_ensemble"]])

        for name in models.keys():
            topm = preds.sort_values(f"proba_{name}", ascending=False).head(top_k)
            print(f"=== {name} Top-{top_k} ===")
            print(topm[keep_cols + [f"proba_{name}"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", default=None, help="Override input CSV path")
    ap.add_argument("--event", default=None, help="Override event code")
    args = ap.parse_args()
    main(args.config, input_override=args.input, event_code_override=args.event)