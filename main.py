import gzip
import io
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIG
# ============================================================

MAX_DECOMPRESSED_BYTES = 200 * 1024 * 1024   # 200 MB safety ceiling
MODEL_SELECTION_MAX_ROWS = 30_000
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Categorical protection against one-hot explosions
MAX_CATEGORY_LEVELS = 100
MIN_CATEGORY_FREQUENCY = 10

app = FastAPI(
    title="John Data Analyzer API",
    version="2.0.0",
    description="Adaptive no-code ML training API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://johnbarandica.com",
        "https://www.johnbarandica.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def detect_problem_type(y: pd.Series) -> str:
    unique_count = y.nunique(dropna=True)
    unique_ratio = unique_count / max(len(y), 1)

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    if unique_count <= 20 or unique_ratio <= 0.05:
        return "classification"

    return "regression"


def optimize_dataframe_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loss-conscious memory optimization:
    - downcast integers and floating-point values where practical
    - convert low-cardinality object/string columns to category
    """
    before = int(df.memory_usage(deep=True).sum())
    changed = []

    for col in df.columns:
        s = df[col]

        try:
            if pd.api.types.is_integer_dtype(s):
                new_s = pd.to_numeric(s, downcast="integer")
                if new_s.dtype != s.dtype:
                    df[col] = new_s
                    changed.append(col)

            elif pd.api.types.is_float_dtype(s):
                # float32 is generally sufficient for tabular ML and
                # substantially lowers memory pressure.
                new_s = pd.to_numeric(s, downcast="float")
                if new_s.dtype != s.dtype:
                    df[col] = new_s
                    changed.append(col)

            elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                non_null = s.dropna()
                if len(non_null):
                    unique = non_null.nunique()
                    ratio = unique / len(non_null)
                    if unique <= 500 and ratio <= 0.50:
                        df[col] = s.astype("category")
                        changed.append(col)
        except Exception:
            # Optimization must never make the request fail.
            continue

    after = int(df.memory_usage(deep=True).sum())
    reduction_pct = 0.0 if before == 0 else (1 - after / before) * 100

    return df, {
        "memory_before_mb": round(before / 1024 / 1024, 2),
        "memory_after_mb": round(after / 1024 / 1024, 2),
        "memory_reduction_pct": round(max(0.0, reduction_pct), 1),
        "optimized_columns": len(changed),
    }


def detect_unusable_features(X: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Drop only features that are very unlikely to be useful in this generic
    tabular pipeline:
      - constants / all-null
      - near-unique text identifiers/free-text columns
    """
    constants = []
    identifier_like = []
    free_text = []

    id_tokens = ("id", "uuid", "guid", "email", "url", "phone", "account", "ssn")

    for col in X.columns:
        s = X[col]
        non_null = s.dropna()
        nunique = non_null.nunique()

        if nunique <= 1:
            constants.append(col)
            continue

        is_text = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        )

        if not is_text or len(non_null) < 20:
            continue

        ratio = nunique / len(non_null)
        col_lower = str(col).lower()

        if ratio >= 0.98 and any(token in col_lower for token in id_tokens):
            identifier_like.append(col)
            continue

        # Generic free-text / near-unique string protection.
        if ratio >= 0.95 and nunique > MAX_CATEGORY_LEVELS:
            avg_len = non_null.astype(str).str.len().mean()
            if avg_len >= 18:
                free_text.append(col)

    return constants, identifier_like, free_text


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    max_categories=MAX_CATEGORY_LEVELS,
                    sparse_output=True,
                ),
            ),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise HTTPException(status_code=400, detail="No usable predictor columns remain.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


def representative_sample(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    max_rows: int,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    if len(X) <= max_rows:
        return X, y, "full_training_split"

    if problem_type == "classification":
        try:
            X_s, _, y_s, _ = train_test_split(
                X,
                y,
                train_size=max_rows,
                random_state=RANDOM_STATE,
                stratify=y,
            )
            return X_s, y_s, "stratified"
        except ValueError:
            pass

    else:
        # Preserve the target distribution for regression when possible.
        try:
            bins = pd.qcut(y, q=min(10, y.nunique()), duplicates="drop")
            X_s, _, y_s, _ = train_test_split(
                X,
                y,
                train_size=max_rows,
                random_state=RANDOM_STATE,
                stratify=bins,
            )
            return X_s, y_s, "target_quantile_stratified"
        except Exception:
            pass

    chosen = X.sample(n=max_rows, random_state=RANDOM_STATE).index
    return X.loc[chosen], y.loc[chosen], "random"


def build_models(problem_type: str) -> Dict[str, Any]:
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1200,
                solver="saga",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        }

    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def classification_scoring(y: pd.Series):
    n_classes = y.nunique()

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    # ROC-AUC can be fragile for tiny classes in CV, so enable only
    # when each class is sufficiently represented.
    counts = y.value_counts()
    if n_classes == 2 and counts.min() >= 5:
        scoring["roc_auc"] = "roc_auc"
    elif 2 < n_classes <= 20 and counts.min() >= 5:
        scoring["roc_auc"] = "roc_auc_ovr_weighted"

    return scoring


def cv_strategy(y: pd.Series, problem_type: str):
    # Adaptive fold count reduces compute on large model-selection samples.
    if problem_type == "classification":
        min_class = int(y.value_counts().min())
        folds = min(5 if len(y) <= 15_000 else 3, min_class)
        folds = max(2, folds)
        return StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=RANDOM_STATE
        ), folds

    folds = 5 if len(y) <= 15_000 else 3
    return KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE), folds


def evaluate_holdout(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
) -> Dict[str, Any]:
    pred = pipeline.predict(X_test)

    if problem_type == "classification":
        result = {
            "accuracy": safe_float(accuracy_score(y_test, pred)),
            "precision": safe_float(
                precision_score(y_test, pred, average="weighted", zero_division=0)
            ),
            "recall": safe_float(
                recall_score(y_test, pred, average="weighted", zero_division=0)
            ),
            "f1": safe_float(
                f1_score(y_test, pred, average="weighted", zero_division=0)
            ),
            "roc_auc": None,
        }

        try:
            if hasattr(pipeline, "predict_proba"):
                prob = pipeline.predict_proba(X_test)
                n_classes = y_test.nunique()
                if n_classes == 2:
                    result["roc_auc"] = safe_float(
                        roc_auc_score(y_test, prob[:, 1])
                    )
                elif n_classes <= 20:
                    result["roc_auc"] = safe_float(
                        roc_auc_score(
                            y_test, prob, multi_class="ovr", average="weighted"
                        )
                    )
        except Exception:
            pass

        return result

    return {
        "r2": safe_float(r2_score(y_test, pred)),
        "mae": safe_float(mean_absolute_error(y_test, pred)),
        "rmse": safe_float(math.sqrt(mean_squared_error(y_test, pred))),
    }


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "service": "data-analyzer-api", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    target: str = Form(...),
    compressed: Optional[str] = Form(None),
    original_name: Optional[str] = Form(None),
):
    started = time.time()

    try:
        raw = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}")

    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Browser may gzip the CSV before upload. This is lossless.
    if compressed == "gzip":
        try:
            raw = gzip.decompress(raw)
        except Exception:
            raise HTTPException(status_code=400, detail="The compressed CSV could not be decompressed.")

    if len(raw) > MAX_DECOMPRESSED_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Dataset is too large after decompression for the current service tier.",
        )

    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", low_memory=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"CSV parsing failed: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV parsing failed: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV contains no usable rows.")

    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f'Target column "{target}" was not found.')

    original_rows = len(df)
    original_columns = len(df.columns)

    # Exact duplicate removal is generally safe for ordinary supervised
    # tabular ML and reduces needless compute. We report it transparently.
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = before_dedup - len(df)

    # Remove rows with no target. Predictors may still contain missing values;
    # those are imputed inside each CV fold, avoiding leakage.
    missing_target_rows = int(df[target].isna().sum())
    df = df[df[target].notna()].copy()

    if len(df) < 20:
        raise HTTPException(
            status_code=400,
            detail="At least 20 rows with a non-missing target are required.",
        )

    df, memory_info = optimize_dataframe_dtypes(df)

    y = df[target]
    X = df.drop(columns=[target])

    constants, identifier_like, free_text = detect_unusable_features(X)
    dropped_features = constants + identifier_like + free_text
    if dropped_features:
        X = X.drop(columns=dropped_features, errors="ignore")

    if X.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="No usable predictor columns remain after data-quality checks.",
        )

    problem_type = detect_problem_type(y)

    if problem_type == "classification":
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            raise HTTPException(
                status_code=400,
                detail="Classification requires at least two target classes.",
            )
        if class_counts.min() < 2:
            raise HTTPException(
                status_code=400,
                detail="Every target class needs at least two examples.",
            )

    # Hold out untouched data BEFORE model-selection sampling.
    try:
        stratify = y if problem_type == "classification" else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    # Compare candidate models on a representative subset for large datasets.
    X_select, y_select, sample_method = representative_sample(
        X_train_full,
        y_train_full,
        problem_type,
        MODEL_SELECTION_MAX_ROWS,
    )

    preprocessor = make_preprocessor(X_select)
    models = build_models(problem_type)
    cv, folds = cv_strategy(y_select, problem_type)

    results: List[Dict[str, Any]] = []
    fitted_model_defs = {}

    if problem_type == "classification":
        scoring = classification_scoring(y_select)
        selection_metric = "F1 (weighted)"

        for name, estimator in models.items():
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            try:
                scores = cross_validate(
                    pipeline,
                    X_select,
                    y_select,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,  # Avoid nested process explosions on Cloud Run.
                    error_score="raise",
                )

                row = {
                    "model": name,
                    "accuracy": safe_float(np.mean(scores["test_accuracy"])),
                    "precision": safe_float(np.mean(scores["test_precision"])),
                    "recall": safe_float(np.mean(scores["test_recall"])),
                    "f1": safe_float(np.mean(scores["test_f1"])),
                    "f1_std": safe_float(np.std(scores["test_f1"])),
                    "roc_auc": (
                        safe_float(np.mean(scores["test_roc_auc"]))
                        if "test_roc_auc" in scores else None
                    ),
                    "cv_folds": folds,
                }
                results.append(row)
                fitted_model_defs[name] = estimator
            except Exception as exc:
                results.append({
                    "model": name,
                    "error": f"{type(exc).__name__}: {str(exc)[:180]}",
                    "cv_folds": folds,
                })

        valid = [r for r in results if r.get("f1") is not None]
        if not valid:
            raise HTTPException(
                status_code=422,
                detail="None of the classification models could be trained on this dataset.",
            )
        best = max(valid, key=lambda r: r["f1"])

    else:
        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }
        selection_metric = "R²"

        for name, estimator in models.items():
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            try:
                scores = cross_validate(
                    pipeline,
                    X_select,
                    y_select,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                    error_score="raise",
                )

                row = {
                    "model": name,
                    "r2": safe_float(np.mean(scores["test_r2"])),
                    "r2_std": safe_float(np.std(scores["test_r2"])),
                    "mae": safe_float(-np.mean(scores["test_mae"])),
                    "rmse": safe_float(-np.mean(scores["test_rmse"])),
                    "cv_folds": folds,
                }
                results.append(row)
                fitted_model_defs[name] = estimator
            except Exception as exc:
                results.append({
                    "model": name,
                    "error": f"{type(exc).__name__}: {str(exc)[:180]}",
                    "cv_folds": folds,
                })

        valid = [r for r in results if r.get("r2") is not None]
        if not valid:
            raise HTTPException(
                status_code=422,
                detail="None of the regression models could be trained on this dataset.",
            )
        best = max(valid, key=lambda r: r["r2"])

    best_model_name = best["model"]

    # Retrain ONLY the winner on the full training split, then evaluate it
    # against the untouched holdout set.
    final_preprocessor = make_preprocessor(X_train_full)
    final_pipeline = Pipeline([
        ("preprocessor", final_preprocessor),
        ("model", fitted_model_defs[best_model_name]),
    ])
    final_pipeline.fit(X_train_full, y_train_full)
    holdout_metrics = evaluate_holdout(
        final_pipeline, X_test, y_test, problem_type
    )

    elapsed = round(time.time() - started, 2)

    return {
        "api_version": "2.0.0",
        "problem_type": problem_type,
        "target": target,
        "best_model": best_model_name,
        "selection_metric": selection_metric,
        "models": results,
        "rows_used": int(len(df)),
        "features_used": int(X.shape[1]),
        "holdout_rows": int(len(X_test)),
        "model_selection_rows": int(len(X_select)),
        "final_training_rows": int(len(X_train_full)),
        "final_test_metrics": holdout_metrics,
        "optimization": {
            **memory_info,
            "original_rows": int(original_rows),
            "original_columns": int(original_columns),
            "duplicate_rows_removed": int(duplicate_rows_removed),
            "missing_target_rows_removed": int(missing_target_rows),
            "sampling_applied": bool(len(X_select) < len(X_train_full)),
            "sampling_method": sample_method,
            "dropped_constant_features": constants,
            "dropped_identifier_features": identifier_like,
            "dropped_free_text_features": free_text,
            "categorical_max_levels": MAX_CATEGORY_LEVELS,
            "categorical_min_frequency": MIN_CATEGORY_FREQUENCY,
        },
        "processing_seconds": elapsed,
        "upload_compression": "gzip" if compressed == "gzip" else "none",
        "source_filename": original_name or file.filename,
    }
