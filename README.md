<!-- Badges -->
[![License](https://img.shields.io/github/license/Charitha-Jeewanka/F1-Winner-Predictions)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Last commit](https://img.shields.io/github/last-commit/Charitha-Jeewanka/F1-Winner-Predictions/master)](https://github.com/Charitha-Jeewanka/F1-Winner-Predictions/commits/master)

# F1 Winner prediction Model

Predict the **race winner** before lights out using qualifying data and other pre‑race features. This repo provides a **config‑driven** pipeline for:

* **Training** multiple models (XGBoost, CatBoost, LightGBM, RandomForest, LogisticRegression)
* **Evaluation** on a held‑out test split with race‑aware metrics (Top‑K per race)
* **Inference** from the weekend’s qualifying sheet (CSV) to produce per‑driver win probabilities and ensemble picks
* **MLflow logging** for training, evaluation, and inference (metrics, params, artifacts, models)

> Key design goals: **no label leakage**, **grouped splits by race**, **class imbalance handling**, **reproducibility**, and **clean MLflow lineage**.

---

## Repository structure

```
.
├─ configs/
│  └─ config.yaml
├─ data/
│  ├─ processed/
│  │  └─ race_dataset_all.csv
│  └─ inference/
│     └─ this_weekend_quali.csv
├─ artifacts/
│  ├─ models/                 # one folder per model (saved pipelines + metrics)
│  ├─ reports/                # leaderboards, predictions, plots
│  └─ mlruns/ (optional)      # MLflow local tracking store if using file: URI
├─ pipelines/
│  ├─ training_pipeline.py
│  ├─ evaluation_pipeline.py
│  └─ inference_pipeline.py
├─ src/
│  ├─ metrics/
│  │  └─ race_metrics.py
│  └─ models/
│     └─ registry.py
└─ utils/
   ├─ data.py
   ├─ inference.py
   └─ mlflow_logger.py
```

---

## Setup

### Requirements

* Python 3.10+ (recommended 3.11)
* `pip install -r requirements.txt`

`requirements.txt` (core):

```
pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
pyyaml
joblib
matplotlib
mlflow
```

### Quickstart

```bash
# 1) Train all enabled models
python -m pipelines.training_pipeline --config configs/config.yaml

# 2) Evaluate on the held-out test split
python -m pipelines.evaluation_pipeline --config configs/config.yaml

# 3) Infer from this weekend's qualifying CSV
python -m pipelines.inference_pipeline --config configs/config.yaml \
  --input data/inference/this_weekend_quali.csv --event 2025_12

# (Optional) Start MLflow UI on local file store
mlflow ui --backend-store-uri artifacts/mlruns --port 5000
```

---

## Data & Target

* **Input dataset:** `data/processed/race_dataset_all.csv`
* **Group key:** `event_code` (one GP/weekend)
* **Target:** `is_winner` (created automatically if missing as `finish_pos == 1`)
* **Leakage guard:** columns that contain post‑race info are **dropped** before modeling (see `configs/config.yaml`) such as `finish_pos`, `pos_gain`, `points`, `laps_completed`, `race_gap_s`.

---

## Configuration (`configs/config.yaml`)

```yaml
random_seed: 42

data_paths:
  dataset: "data/processed/race_dataset_all.csv"
  artifacts_dir: "artifacts"
  models_dir: "artifacts/models"
  reports_dir: "artifacts/reports"

columns:
  target: "is_winner"
  finish_pos: "finish_pos"
  group: "event_code"
  leakage_guard: ["finish_pos", "pos_gain", "points", "laps_completed", "race_gap_s"]
  categorical_hints: ["Abbreviation", "team", "status"]
  drop_columns: []

split:
  test_size: 0.15
  val_size: 0.15

models:
  enabled: ["xgboost", "catboost", "lightgbm", "random_forest", "logistic_regression"]
  xgboost:   { params: { n_estimators: 500, learning_rate: 0.05, max_depth: 6, subsample: 0.8, colsample_bytree: 0.8, tree_method: hist, eval_metric: logloss } }
  catboost:  { params: { iterations: 1000, depth: 6, learning_rate: 0.05, loss_function: Logloss, verbose: 200, allow_writing_files: false } }
  lightgbm:  { params: { n_estimators: 800, learning_rate: 0.05, num_leaves: 63, subsample: 0.8, colsample_bytree: 0.8, objective: binary } }
  random_forest: { params: { n_estimators: 600, max_depth: null, n_jobs: -1, random_state: 42 } }
  logistic_regression: { params: { penalty: l2, C: 1.0, solver: liblinear, max_iter: 2000 } }

metrics:
  ks: [1, 3]       # race-level Top-K hit rates
  threshold: 0.5   # classification threshold
  plots: true

mlflow:
  enabled: true
  tracking_uri: "file:artifacts/mlruns"   # or http://127.0.0.1:5000
  experiment: "F1 Winner Model"
  nested: true
  tags: { project: "f1-winner-model" }

# Optional probability calibration
calibration:
  enabled: true
  method: sigmoid
  models: ["xgboost", "lightgbm", "random_forest", "catboost"]

# Inference block
inference:
  input_csv: "data/inference/this_weekend_quali.csv"
  event_code: "this_event"
  ensemble: "weighted"       # mean | weighted | rank_rrf
  top_k: 3
  models: null                # or e.g. ["catboost", "xgboost"]
```

---

## Training

* Grouped split (by `event_code`) to avoid cross‑race leakage.
* Class imbalance handled via `scale_pos_weight` (boosters) / `class_weight` (sklearn).
* Preprocessing pipeline:

  * **Categoricals:** Impute → OneHotEncoder (`handle_unknown=ignore`) with version‑proof sparse/dense flag
  * **Numerics:** Impute (median), scaled only for Logistic Regression
  * **VarianceThreshold:** drops constant columns after OHE
* Artifacts per model under `artifacts/models/<model_name>/`:

  * `model.joblib` — full sklearn pipeline (preprocess → variance → classifier)
  * `metrics_val.json` — validation metrics
  * `val_predictions.csv` — per‑sample outputs
* MLflow:

  * Parent run `phase=training` → child runs per model `stage=train`
  * Logs params, `val_*` metrics, artifacts, and a serialized sklearn model (`sklearn_model`)

Run:

```bash
python -m pipelines.training_pipeline --config configs/config.yaml
```

---

## Evaluation

* Uses the **held‑out** test split persisted by training (`artifacts/test_split.json`).
* Logs **classification metrics** (Accuracy, Precision, Recall, F1, ROC‑AUC, PR‑AUC, Log Loss, Brier) and **race‑level metrics** (`race_top1`, `race_top3`).
* Saves per‑model plots: ROC & Confusion Matrix.
* Writes `artifacts/reports/test_leaderboard.csv` and logs everything to MLflow under `phase=evaluation`.

Run:

```bash
python -m pipelines.evaluation_pipeline --config configs/config.yaml
```

---

## Inference

Given the qualifying results CSV, the pipeline:

1. Parses Q1/Q2/Q3 into seconds (e.g., `1:18.792` → `78.792`)
2. Aligns columns to the **training schema** (missing numeric → NaN; missing categorical → "missing")
3. Scores all saved models and builds an **ensemble** (`mean`, `weighted` by validation `race_top1`, or **`rank_rrf`**)
4. Outputs per‑driver probabilities and **Top‑K** predictions

Example CSV (built from a quali sheet):

```csv
event_code,grid_pos,driver_number,driver,team,Abbreviation,Q1,Q2,Q3,LAPS
2025_12,1,1,Max Verstappen,Red Bull Racing,VER,1:19.455,1:19.140,1:18.792,18
2025_12,2,4,Lando Norris,McLaren,NOR,1:19.517,1:19.293,1:18.869,21
...
```

Run:

```bash
python -m pipelines.inference_pipeline --config configs/config.yaml \
  --input data/inference/this_weekend_quali.csv --event 2025_12
```

Outputs:

* `artifacts/reports/predictions/<event_code>/predictions.csv` — per‑model and ensemble probabilities + ranks
* Console: Ensemble Top‑K and per‑model Top‑K tables
* MLflow: `phase=inference`, artifacts under `predictions/<event_code>/`

**Ensemble modes**

* `mean` — arithmetic mean of probabilities
* `weighted` — weighted by each model’s validation `race_top1`
* `rank_rrf` — robust rank fusion using reciprocal ranks (good when models are differently calibrated)

---

## Metrics

* **Classification:** `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `log_loss`, `brier`
* **Race‑level Top‑K:** For each race (group), sort drivers by predicted win probability and check if the true winner is in the Top‑K. Report the mean across races as `race_topK`.

These **race‑level** metrics are most aligned with the problem of picking a winner.

---

## MLflow

* Configure in `configs/config.yaml` (`mlflow.enabled`, `tracking_uri`, `experiment`).
* Three phases are logged:

  * `training` → child runs per model with `val_*`
  * `evaluation` → child runs per model with `test_*`
  * `inference` → one run per event (predictions + Top‑K summary + ensemble weights)
* To view locally:

```bash
mlflow ui --backend-store-uri artifacts/mlruns --port 5000
```

---

## Troubleshooting

**OneHotEncoder `sparse` error**
Use the version‑proof encoder in `utils/data.py` (switches between `sparse_output` and `sparse`).

**LightGBM: `No further splits with positive gain`**
Harmless but suggests underfitting—relax split constraints (e.g., `min_child_samples`, `min_sum_hessian_in_leaf`, `num_leaves`) and drop constant columns (we use `VarianceThreshold`).

**`X does not have valid feature names` (LightGBM)**
A warning when predicting via pipeline—safe to ignore. Can be filtered in the eval script.

**LogisticRegression NaN error**
Resolved via imputers in the preprocessing pipeline (numeric median, categorical most‑frequent).

**MLflow: invalid model name / `artifact_path` deprecated**
The helper uses `name=` and sanitizes names. Log models as `sklearn_model` inside each child run.

**Calibration API differences**
`CalibratedClassifierCV` may use `estimator=` (new) or `base_estimator=` (old). The training script tries both.

---

## Extending

* **Add models:** register a new key in `src/models/registry.py` and update `configs/config.yaml`.
* **Ranking models:** add LightGBM’s `LGBMRanker` (`lambdarank`) with group=`event_code` and log NDCG\@K/MRR.
* **Feature engineering:** enrich with strictly pre‑race features (team/driver form, track history, weather expectation, PU/constructor, teammate deltas).
* **Calibration:** toggle in `configs/config.yaml` per model (`sigmoid` or `isotonic`).

---
