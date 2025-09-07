from __future__ import annotations
from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except Exception:  
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  
    LGBMClassifier = None


SUPPORTED = {"xgboost", "catboost", "lightgbm", "random_forest", "logistic_regression"}


def build_estimator(model_name: str, params: Dict[str, Any], pos_weight: float, random_state: int):
    name = model_name.lower()
    if name not in SUPPORTED:
        raise ValueError(f"Unknown model: {model_name}")

    if name == "random_forest":
        p = {**{"class_weight": "balanced", "random_state": random_state}, **(params or {})}
        return RandomForestClassifier(**p)

    if name == "logistic_regression":
        class_weight = {0: 1.0, 1: float(pos_weight)}
        p = {**{"class_weight": class_weight, "random_state": random_state}, **(params or {})}
        return LogisticRegression(**p)

    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. pip install xgboost")
        p = {**{"random_state": random_state, "n_jobs": -1}, **(params or {})}
        p.setdefault("scale_pos_weight", float(pos_weight))
        return XGBClassifier(**p)

    if name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed. pip install catboost")
        p = {**{"random_seed": random_state, "thread_count": -1}, **(params or {})}
        p.setdefault("class_weights", [1.0, float(pos_weight)])
        return CatBoostClassifier(**p)

    if name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed. pip install lightgbm")
        p = {**{"random_state": random_state, "n_jobs": -1}, **(params or {})}
        p.setdefault("scale_pos_weight", float(pos_weight))
        return LGBMClassifier(**p)

    raise ValueError(f"Unhandled model: {model_name}")