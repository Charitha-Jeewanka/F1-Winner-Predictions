from __future__ import annotations
from typing import Iterable, Tuple, Set

import inspect
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_and_prepare(
    csv_path,
    target_col: str,
    finish_pos_col: str,
    group_col: str,
    leakage_guard: Iterable[str] = (),
    drop_columns: Iterable[str] = (),
):
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        if finish_pos_col not in df.columns:
            raise ValueError(f"Target '{target_col}' missing and finish_pos '{finish_pos_col}' not available to derive it.")
        df[target_col] = (df[finish_pos_col].astype(float) == 1).astype(int)

    cols_to_drop = [c for c in leakage_guard if c in df.columns]
    cols_to_drop.extend([c for c in drop_columns if c in df.columns])
    cols_to_drop = list(dict.fromkeys(cols_to_drop))  

    keep_cols = [c for c in df.columns if c not in cols_to_drop]
    df = df[keep_cols]

    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")

    return df, target_col, group_col


def infer_coltypes(X: pd.DataFrame, cat_hint_names: Set[str] | None = None):
    cat_hint_names = set(cat_hint_names or [])
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or c in cat_hint_names]
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return num_cols, cat_cols


def grouped_train_val_test_split(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    rng: np.random.RandomState | None = None,
):
    rng = rng or np.random.RandomState(42)

    groups = df[group_col]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rng.randint(0, 1_000_000))
    idx_trainval, idx_test = next(gss1.split(X, y, groups=groups))

    X_trainval, y_trainval, g_trainval = X.iloc[idx_trainval], y.iloc[idx_trainval], groups.iloc[idx_trainval]
    X_test, y_test, g_test = X.iloc[idx_test], y.iloc[idx_test], groups.iloc[idx_test]

    
    rel_val_size = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=rng.randint(0, 1_000_000))
    idx_train, idx_val = next(gss2.split(X_trainval, y_trainval, groups=g_trainval))

    X_train, y_train, g_train = X_trainval.iloc[idx_train], y_trainval.iloc[idx_train], g_trainval.iloc[idx_train]
    X_val, y_val, g_val = X_trainval.iloc[idx_val], y_trainval.iloc[idx_val], g_trainval.iloc[idx_val]

    return (X_train, y_train, g_train), (X_val, y_val, g_val), (X_test, y_test, g_test)



def _make_ohe(sparse: bool):
    params = {"handle_unknown": "ignore"}
    sig = inspect.signature(OneHotEncoder)
    if "sparse_output" in sig.parameters:   # sklearn >= 1.2
        params["sparse_output"] = sparse
    else:                                   # sklearn < 1.2
        params["sparse"] = sparse
    return OneHotEncoder(**params)

def build_preprocessor(model_name: str, num_cols, cat_cols):
    model_name = model_name.lower()
    ohe = _make_ohe(sparse=False)

    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()) if model_name == "logistic_regression" else ("noop", "passthrough")
    ])

    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, list(cat_cols)),
            ("num", num_pipe, list(num_cols)),
        ],
        remainder="drop",
    )
    return preproc

