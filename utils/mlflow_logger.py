from __future__ import annotations
from pathlib import Path
import json
import os
import subprocess
from contextlib import contextmanager

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_OK = True
except Exception:
    MLFLOW_OK = False


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def setup(cfg: dict):
    cfg = cfg or {}
    if not cfg.get("enabled", False) or not MLFLOW_OK:
        return False
    mlflow.set_tracking_uri(cfg.get("tracking_uri", "file:mlruns"))
    mlflow.set_experiment(cfg.get("experiment", "default"))
    return True


@contextmanager
def parent_run(cfg: dict, phase: str, run_name: str | None = None, tags: dict | None = None):
    if not setup(cfg):
        yield None
        return
    base_tags = {"phase": phase, **(cfg.get("tags") or {})}
    gh = _git_hash()
    if gh:
        base_tags["git_hash"] = gh
    if tags:
        base_tags.update(tags)
    with mlflow.start_run(run_name=run_name or f"{phase}", tags=base_tags) as run:
        yield run


@contextmanager
def child_run(model_name: str, stage: str = "train", tags: dict | None = None):
    if not MLFLOW_OK:
        yield None
        return
    t = {"model": model_name, "stage": stage}
    if tags:
        t.update(tags)
    with mlflow.start_run(run_name=f"{stage}:{model_name}", nested=True, tags=t) as run:
        yield run


def log_params(params: dict):
    if MLFLOW_OK:
        mlflow.log_params(params or {})


def log_metrics(metrics: dict, step: int | None = None):
    if MLFLOW_OK:
        mlflow.log_metrics({k: float(v) for k, v in (metrics or {}).items()}, step=step)


def log_artifacts(path: Path | str, artifact_path: str | None = None):
    if MLFLOW_OK:
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def log_dict(obj: dict, out_name: str):
    if not MLFLOW_OK:
        return
    mlflow.log_dict(obj, out_name)


def log_model_sklearn(model, subname: str = "sklearn_model", registered_model_name: str | None = None):
    if not MLFLOW_OK:
        return
    import inspect
    safe = (subname or "sklearn_model")
    for bad in ["/", ":", ".", "%", '"', "'"]:
        safe = safe.replace(bad, "_")

    fn = mlflow.sklearn.log_model
    sig = inspect.signature(fn)
    if "name" in sig.parameters:  
        fn(model, name=safe, registered_model_name=registered_model_name)
    else:  # legacy API
        fn(model, artifact_path=safe, registered_model_name=registered_model_name)
