from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import metrics as skm


def compute_classification_metrics(y_true, y_pred, y_proba):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    out = {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(skm.roc_auc_score(y_true, y_proba)),
        "pr_auc": float(skm.average_precision_score(y_true, y_proba)),
        "log_loss": float(skm.log_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6))),
        "brier": float(skm.brier_score_loss(y_true, y_proba)),
    }
    return out


def compute_race_topk_metrics(y_true, y_proba, groups, ks=(1, 3)):
    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_proba": np.asarray(y_proba),
        "group": np.asarray(groups),
    })

    results = {}
    grouped = df.groupby("group", sort=False)

    for k in ks:
        hits = 0
        total = 0
        for _, g in grouped:
            topk_idx = g["y_proba"].nlargest(k).index
            true_idx = g.index[g["y_true"] == 1]
            if len(true_idx) == 0:
                continue
            hit = len(set(topk_idx).intersection(set(true_idx))) > 0
            hits += int(hit)
            total += 1
        results[f"race_top{k}"] = float(hits / max(total, 1))

    return results