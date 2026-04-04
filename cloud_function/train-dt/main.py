# train-dt/main.py
# Trains an improved model with Optuna hyperparameter tuning.
# Outputs 3 artifacts to GCS structured/preds/<TIMESTAMP>/:
#   - preds.csv
#   - permutation_importance.csv
#   - pdp_top3.png

import os, io, json, logging, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Cloud Functions
import matplotlib.pyplot as plt

from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- ENV ----
PROJECT_ID    = os.getenv("PROJECT_ID", "")
GCS_BUCKET    = os.getenv("GCS_BUCKET", "")
DATA_KEY      = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm_V1.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "structured/preds")
TIMEZONE      = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL     = os.getenv("LOG_LEVEL", "INFO")
N_TRIALS      = int(os.getenv("OPTUNA_TRIALS", "30"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


# -------------------- GCS HELPERS --------------------
def _read_csv_from_gcs(client, bucket, key):
    blob = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

def _upload_bytes(client, bucket, key, data: bytes, content_type="text/plain"):
    client.bucket(bucket).blob(key).upload_from_string(
        data, content_type=content_type
    )

def _upload_df_csv(client, bucket, key, df):
    _upload_bytes(
        client, bucket, key,
        df.to_csv(index=False).encode(),
        "text/csv"
    )

def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# -------------------- FEATURE ENGINEERING --------------------
def _engineer_features(df):
    current_year = 2026

    df['price']    = pd.to_numeric(df['price'],    errors='coerce')
    df['mileage']  = pd.to_numeric(df['mileage'],  errors='coerce')
    df['year']     = pd.to_numeric(df['year'],     errors='coerce')
    df['cylinder'] = pd.to_numeric(df['cylinder'], errors='coerce')

    # Vehicle age
    df['vehicle_age'] = current_year - df['year']

    # Age bin
    df['age_bin'] = pd.cut(
        df['vehicle_age'],
        bins=[0, 5, 10, 20, float('inf')],
        labels=['nearly_new', 'recent', 'aging', 'old']
    ).astype(str)

    # Mileage bin
    df['mileage_bin'] = pd.cut(
        df['mileage'],
        bins=[0, 50_000, 100_000, float('inf')],
        labels=['low', 'medium', 'high']
    ).astype(str)

    # Remove bad prices
    df = df[
        df['price'].notna() &
        (df['price'] > 500) &
        (df['price'] < 200_000)
    ]
    return df


# -------------------- MAIN TRAINING FUNCTION --------------------
def run_once(dry_run=False, n_trials=N_TRIALS):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    # Validate required columns
    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = _engineer_features(df)

    # ---- Date-based train/holdout split ----
    df['scraped_at'] = pd.to_datetime(df['scraped_at'], utc=True, errors='coerce')
    df['date_local'] = df['scraped_at'].dt.tz_convert(TIMEZONE).dt.date

    unique_dates = sorted(df['date_local'].dropna().unique())
    if len(unique_dates) < 2:
        return {
            "status": "noop",
            "reason": "need at least 2 dates",
            "dates":  [str(d) for d in unique_dates]
        }

    today      = unique_dates[-1]
    train_df   = df[df['date_local'] <  today].copy()
    holdout_df = df[df['date_local'] == today].copy()

    logging.info("Train rows: %d | Holdout rows: %d",
                 len(train_df), len(holdout_df))

    if len(train_df) < 40:
        return {
            "status":     "noop",
            "reason":     "too few training rows",
            "train_rows": int(len(train_df))
        }

    # ---- Features ----
    all_cat_cols = ['make', 'model', 'transmission', 'fuel_type',
                    'mileage_bin', 'age_bin', 'state']
    all_num_cols = ['year', 'mileage', 'vehicle_age', 'cylinder']

    cat_cols = [c for c in all_cat_cols if c in train_df.columns and train_df[c].notna().any()]
    num_cols = [c for c in all_num_cols if c in train_df.columns and train_df[c].notna().any()]

    features = cat_cols + num_cols
    target   = 'price'

    X_train = train_df[features]
    y_train = train_df[target]

    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols),
    ])

    # ---- Optuna tuning ----
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        }
        pipe = Pipeline([
            ('pre',   pre),
            ('model', GradientBoostingRegressor(**params, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        return mean_absolute_error(y_train, pipe.predict(X_train))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    logging.info("Best params: %s", best_params)

    # ---- Train final model with best params ----
    best_pipe = Pipeline([
        ('pre',   pre),
        ('model', GradientBoostingRegressor(**best_params, random_state=42))
    ])
    best_pipe.fit(X_train, y_train)

    # ---- Evaluate on holdout ----
    X_h   = holdout_df[features]
    y_h   = holdout_df[target]
    y_hat = best_pipe.predict(X_h)

    mae  = float(mean_absolute_error(y_h, y_hat))
    mape = float(mean_absolute_percentage_error(y_h, y_hat) * 100)
    rmse = float(np.sqrt(((y_h - y_hat) ** 2).mean()))
    bias = float((y_hat - y_h).mean())

    logging.info("MAE=%.0f MAPE=%.1f%% RMSE=%.0f Bias=%.0f",
                 mae, mape, rmse, bias)

    # ---- Output path — YYYYMMDDHH to match yml expectations ----
    now_utc  = pd.Timestamp.utcnow()
    run_ts   = now_utc.strftime('%Y%m%d%H')
    out_base = f"{OUTPUT_PREFIX}/{run_ts}"

    # -------------------- ARTIFACT 1: preds.csv --------------------
    preds_df = holdout_df[['post_id', 'scraped_at', 'make', 'model',
                            'year', 'mileage', 'price']].copy()
    preds_df['predicted_price'] = np.round(y_hat, 2)
    preds_df['error']           = preds_df['predicted_price'] - preds_df['price']
    preds_df['abs_error']       = preds_df['error'].abs()

    if not dry_run:
        _upload_df_csv(
            client, GCS_BUCKET,
            f"{out_base}/preds.csv",
            preds_df
        )
        logging.info("Uploaded preds.csv → gs://%s/%s/preds.csv",
                     GCS_BUCKET, out_base)

    # -------------------- ARTIFACT 2: permutation_importance.csv --------------------
    cat_feature_names = list(
        best_pipe.named_steps['pre']
        .named_transformers_['cat']
        .named_steps['oh']
        .get_feature_names_out(cat_cols)
    )
    all_feature_names = num_cols + cat_feature_names
    X_train_t = best_pipe.named_steps['pre'].transform(X_train)

    perm = permutation_importance(
        best_pipe.named_steps['model'],
        X_train_t,
        y_train,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    imp_df = pd.DataFrame({
        'feature':    all_feature_names,
        'importance': perm.importances_mean
    }).sort_values('importance', ascending=False)

    if not dry_run:
        _upload_df_csv(
            client, GCS_BUCKET,
            f"{out_base}/permutation_importance.csv",
            imp_df
        )
        logging.info("Uploaded permutation_importance.csv → gs://%s/%s/permutation_importance.csv",
                     GCS_BUCKET, out_base)

    # -------------------- ARTIFACT 3: pdp_top3.png --------------------
    # Pick top 3 numeric features by importance
    top3_features = (
        imp_df[imp_df['feature'].isin(num_cols)]
        .head(3)['feature']
        .tolist()
    )
    # Fallback if fewer than 3 numeric features rank highly
    if len(top3_features) < 3:
        top3_features = num_cols[:3]

    top3_idx = [all_feature_names.index(f) for f in top3_features]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    PartialDependenceDisplay.from_estimator(
        best_pipe.named_steps['model'],
        X_train_t,
        features=top3_idx,
        feature_names=all_feature_names,
        ax=axes
    )
    for i, ax in enumerate(axes):
        ax.set_title(f'PDP: {top3_features[i]}')
    plt.suptitle('Partial Dependence Plots — Top 3 Features', y=1.02)
    plt.tight_layout()

    if not dry_run:
        _upload_bytes(
            client, GCS_BUCKET,
            f"{out_base}/pdp_top3.png",
            _fig_to_png_bytes(fig),
            "image/png"
        )
        logging.info("Uploaded pdp_top3.png → gs://%s/%s/pdp_top3.png",
                     GCS_BUCKET, out_base)
    plt.close(fig)

    # ---- Return summary ----
    return {
        "status":        "ok",
        "run_ts":        run_ts,
        "today_local":   str(today),
        "train_rows":    int(len(train_df)),
        "holdout_rows":  int(len(holdout_df)),
        "mae":           round(mae, 2),
        "mape":          round(mape, 2),
        "rmse":          round(rmse, 2),
        "bias":          round(bias, 2),
        "best_params":   best_params,
        "top3_features": top3_features,
        "output_prefix": f"gs://{GCS_BUCKET}/{out_base}",
        "dry_run":       dry_run,
    }


# -------------------- HTTP ENTRY --------------------
def train_dt_http(request):
    try:
        body     = request.get_json(silent=True) or {}
        dry_run  = bool(body.get("dry_run", False))
        n_trials = int(body.get("n_trials", N_TRIALS))
        result   = run_once(dry_run=dry_run, n_trials=n_trials)
        code     = 200 if result.get("status") == "ok" else 204
        return (
            json.dumps(result),
            code,
            {"Content-Type": "application/json"}
        )
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (
            json.dumps({"status": "error", "error": str(e)}),
            500,
            {"Content-Type": "application/json"}
        )