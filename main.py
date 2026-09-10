import gzip
import io
import math
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
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
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# OPTIONAL HIGH-PERFORMANCE MODELS
# ============================================================

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

MAX_DECOMPRESSED_BYTES = 250 * 1024 * 1024
MODEL_SELECTION_MAX_ROWS = 30_000

# Keep generic one-hot features controlled.
MAX_CATEGORY_LEVELS_SMALL = 200
MAX_CATEGORY_LEVELS_LARGE = 100
MIN_CATEGORY_FREQUENCY_SMALL = 3
MIN_CATEGORY_FREQUENCY_LARGE = 10

# AutoML compute budget.
SMALL_DATASET_ROWS = 15_000
TUNING_ITERATIONS_SMALL = 8
TUNING_ITERATIONS_LARGE = 4

# Trust / leakage safeguards
LEAKAGE_UNIQUE_RATIO_THRESHOLD = 0.995
LEAKAGE_SINGLE_FEATURE_ACCURACY = 0.97
LEAKAGE_SINGLE_FEATURE_R2 = 0.95
LEAKAGE_MI_TOP_K = 8

app = FastAPI(
    title="John AutoML API",
    version="3.0.0",
    description="Adaptive no-code AutoML with feature engineering, tuning, ensembling and holdout validation.",
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
# GENERAL HELPERS
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
            continue

    after = int(df.memory_usage(deep=True).sum())
    saved_pct = 0.0 if before == 0 else max(0.0, (1 - after / before) * 100)

    return df, {
        "memory_before_mb": round(before / 1024 / 1024, 2),
        "memory_after_mb": round(after / 1024 / 1024, 2),
        "memory_reduction_pct": round(saved_pct, 1),
        "optimized_columns": len(changed),
    }


# ============================================================
# GENERIC AUTOMATIC FEATURE ENGINEERING
# ============================================================

def _usable_small_category(series: pd.Series, min_coverage=0.20, max_unique=30) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    coverage = len(non_null) / len(series)
    unique = non_null.nunique()
    return coverage >= min_coverage and 2 <= unique <= max_unique


def auto_feature_engineer(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Target-independent feature engineering, so it is safe to perform before
    the train/test split.

    Adds broadly useful signals from:
      - long/high-cardinality text: length, word count, digit count
      - structured text prefixes
      - title-like punctuation patterns when naturally present
      - date/time columns: year, month, day, day-of-week, hour
    """
    X = X.copy()
    created = []

    for col in list(X.columns):
        s = X[col]
        col_name = str(col).lower()

        is_text = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
        )

        # Datetime semantic detection.
        if is_text and any(token in col_name for token in ("date", "time", "timestamp", "created", "updated")):
            parsed = pd.to_datetime(s, errors="coerce", utc=True)
            parse_rate = parsed.notna().mean()

            if parse_rate >= 0.70:
                for suffix, values in {
                    "year": parsed.dt.year,
                    "month": parsed.dt.month,
                    "day": parsed.dt.day,
                    "dow": parsed.dt.dayofweek,
                    "hour": parsed.dt.hour,
                }.items():
                    new_col = f"{col}__{suffix}"
                    X[new_col] = values
                    created.append(new_col)

        if not is_text:
            continue

        as_text = s.astype("string")
        non_null = as_text.dropna()
        if len(non_null) == 0:
            continue

        nunique = non_null.nunique()
        unique_ratio = nunique / len(non_null)
        avg_len = safe_float(non_null.str.len().mean()) or 0.0

        # Salvage information from high-cardinality strings instead of
        # simply discarding them.
        if unique_ratio >= 0.40 or avg_len >= 8:
            new_features = {
                f"{col}__strlen": as_text.str.len().astype("Float32"),
                f"{col}__wordcount": as_text.str.count(r"\S+").astype("Float32"),
                f"{col}__digitcount": as_text.str.count(r"\d").astype("Float32"),
            }

            for name, values in new_features.items():
                X[name] = values
                created.append(name)

        # Generic alphabetic prefix: useful for codes, cabin/deck-like fields,
        # product codes, ticket prefixes, etc.
        prefix = as_text.str.extract(r"^\s*([A-Za-z]{1,12})", expand=False)
        if _usable_small_category(prefix, max_unique=30):
            name = f"{col}__prefix"
            X[name] = prefix
            created.append(name)

        # Generic "title-like" pattern such as "... , Dr." / "... , Mr.".
        # It is only used if the data naturally contains a repeated pattern.
        title_like = as_text.str.extract(r",\s*([^,\.]{1,20})\.", expand=False)
        if _usable_small_category(title_like, max_unique=30):
            name = f"{col}__title"
            X[name] = title_like.str.strip()
            created.append(name)

    return X, created


def detect_unusable_features(
    X: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str]]:
    constants = []
    identifier_like = []
    free_text = []

    id_tokens = (
        "uuid", "guid", "email", "url", "phone", "ssn",
        "account_number", "customer_id", "record_id",
    )

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

        if ratio >= 0.95 and nunique > 100:
            avg_len = safe_float(non_null.astype(str).str.len().mean()) or 0
            if avg_len >= 12:
                free_text.append(col)

    return constants, identifier_like, free_text



# ============================================================
# TRUST / LEAKAGE DIAGNOSTICS
# ============================================================

def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def suspicious_name_relationship(feature: str, target: str) -> bool:
    """
    Flag columns whose names strongly imply direct target encoding.
    Example: target='city', feature='city_name' or 'billing_city'.
    """
    f = _normalized_name(feature)
    t = _normalized_name(target)
    if not f or not t or len(t) < 3:
        return False
    return t in f or f in t


def exact_proxy_score(feature: pd.Series, target: pd.Series) -> Optional[float]:
    """
    For categorical-like values, estimate whether each feature value maps
    almost deterministically to one target value.

    This is target-aware and is used only for diagnostics, not feature creation.
    """
    frame = pd.DataFrame({"f": feature, "y": target}).dropna()
    if len(frame) < 30:
        return None

    # Avoid treating almost-unique identifiers as meaningful predictive features.
    unique_ratio = frame["f"].nunique() / len(frame)
    if unique_ratio >= LEAKAGE_UNIQUE_RATIO_THRESHOLD:
        return None

    grouped = frame.groupby("f", observed=True)["y"]
    mapping = grouped.agg(lambda s: s.value_counts().index[0])
    predicted = frame["f"].map(mapping)

    return safe_float((predicted == frame["y"]).mean())


def numeric_proxy_score(feature: pd.Series, target: pd.Series) -> Optional[float]:
    """
    Simple one-feature linear proxy check for numeric regression.
    Uses squared correlation as a fast warning signal.
    """
    frame = pd.DataFrame({
        "f": pd.to_numeric(feature, errors="coerce"),
        "y": pd.to_numeric(target, errors="coerce"),
    }).dropna()

    if len(frame) < 30:
        return None

    corr = frame["f"].corr(frame["y"])
    if corr is None or pd.isna(corr):
        return None

    return safe_float(float(corr) ** 2)


def leakage_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    problem_type: str,
) -> Dict[str, Any]:
    """
    Identify columns that may directly or indirectly reveal the target.
    The function does NOT automatically delete every flagged feature; it
    returns evidence and a conservative exclusion list for severe cases.
    """
    findings = []
    severe_exclusions = []

    for col in X.columns:
        s = X[col]
        name_flag = suspicious_name_relationship(col, target)

        item = {
            "feature": str(col),
            "name_similarity_flag": bool(name_flag),
            "proxy_score": None,
            "severity": "low",
            "reason": None,
        }

        is_cat = (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or s.nunique(dropna=True) <= 50
        )

        if problem_type == "classification" and is_cat:
            score = exact_proxy_score(s, y)
            item["proxy_score"] = score

            if score is not None and score >= LEAKAGE_SINGLE_FEATURE_ACCURACY:
                item["severity"] = "high"
                item["reason"] = (
                    f"Single-feature mapping reproduces the target with "
                    f"{score:.1%} accuracy."
                )
                severe_exclusions.append(str(col))

        elif problem_type == "regression" and pd.api.types.is_numeric_dtype(s):
            score = numeric_proxy_score(s, y)
            item["proxy_score"] = score

            if score is not None and score >= LEAKAGE_SINGLE_FEATURE_R2:
                item["severity"] = "high"
                item["reason"] = (
                    f"Single numeric feature has approximately R²={score:.3f} "
                    f"against the target."
                )
                severe_exclusions.append(str(col))

        if name_flag and item["severity"] != "high":
            item["severity"] = "medium"
            item["reason"] = (
                "Feature name is very similar to the target name and may encode it."
            )

        if item["severity"] != "low":
            findings.append(item)

    return {
        "findings": findings,
        "severe_exclusions": sorted(set(severe_exclusions)),
        "high_risk_count": sum(1 for x in findings if x["severity"] == "high"),
        "medium_risk_count": sum(1 for x in findings if x["severity"] == "medium"),
    }


def detect_possible_group_column(X: pd.DataFrame) -> Optional[str]:
    """
    Heuristic detection of repeated entity/group identifiers.
    Useful when multiple rows belong to the same customer, patient, device,
    household, session, etc. Random splitting can otherwise leak entity identity.
    """
    tokens = (
        "customer", "client", "patient", "user", "account",
        "household", "device", "session", "member", "subject",
        "person", "company", "store", "site"
    )

    best = None
    best_score = -1.0

    for col in X.columns:
        name = str(col).lower()
        if not any(tok in name for tok in tokens):
            continue

        nunique = X[col].nunique(dropna=True)
        if nunique < 2:
            continue

        ratio = nunique / max(len(X), 1)

        # Strong candidate when identifiers repeat across rows.
        if 0.01 <= ratio <= 0.80:
            score = 1.0 - abs(ratio - 0.30)
            if score > best_score:
                best = str(col)
                best_score = score

    return best


def detect_possible_time_column(X: pd.DataFrame) -> Optional[str]:
    tokens = ("date", "time", "timestamp", "created", "updated", "event")
    best = None
    best_rate = 0.0

    for col in X.columns:
        name = str(col).lower()
        if not any(tok in name for tok in tokens):
            continue

        parsed = pd.to_datetime(X[col], errors="coerce", utc=True)
        rate = parsed.notna().mean()

        if rate >= 0.80 and rate > best_rate:
            best = str(col)
            best_rate = rate

    return best


# ============================================================
# PREPROCESSING
# ============================================================

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    large = len(X) > SMALL_DATASET_ROWS
    max_categories = MAX_CATEGORY_LEVELS_LARGE if large else MAX_CATEGORY_LEVELS_SMALL
    min_frequency = MIN_CATEGORY_FREQUENCY_LARGE if large else MIN_CATEGORY_FREQUENCY_SMALL

    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline([
            # Missingness itself can be predictive, so keep indicator columns.
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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
                    min_frequency=min_frequency,
                    max_categories=max_categories,
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


# ============================================================
# SAMPLING + CV
# ============================================================

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


def cv_strategy(y: pd.Series, problem_type: str, force_three=False):
    if problem_type == "classification":
        min_class = int(y.value_counts().min())
        desired = 3 if force_three or len(y) > SMALL_DATASET_ROWS else 5
        folds = max(2, min(desired, min_class))
        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        ), folds

    folds = 3 if force_three or len(y) > SMALL_DATASET_ROWS else 5
    return KFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    ), folds


def classification_scoring(y: pd.Series):
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    counts = y.value_counts()
    n_classes = y.nunique()

    if counts.min() >= 5:
        if n_classes == 2:
            scoring["roc_auc"] = "roc_auc"
        elif 2 < n_classes <= 20:
            scoring["roc_auc"] = "roc_auc_ovr_weighted"

    return scoring


# ============================================================
# MODEL FACTORY
# ============================================================

def build_models(problem_type: str) -> Dict[str, Any]:
    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1600,
                solver="saga",
                C=1.0,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=350,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                class_weight="balanced_subsample",
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=350,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        }

        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(
                n_estimators=350,
                learning_rate=0.04,
                max_depth=4,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=1.5,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )

        if HAS_LIGHTGBM:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=31,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=-1,
            )

        return models

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=350,
            max_features=0.85,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=350,
            max_features=0.85,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.5,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )

    return models


def tuning_space(model_name: str, problem_type: str) -> Dict[str, List[Any]]:
    if model_name == "Logistic Regression":
        return {
            "model__C": list(np.logspace(-2, 1.3, 16)),
            "model__class_weight": [None, "balanced"],
        }

    if model_name in ("Random Forest", "Extra Trees"):
        return {
            "model__n_estimators": [250, 350, 500, 700],
            "model__max_depth": [None, 6, 10, 16, 24],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__max_features": ["sqrt", "log2", 0.6, 0.85],
        }

    if model_name == "XGBoost":
        return {
            "model__n_estimators": [250, 350, 500, 700],
            "model__learning_rate": [0.02, 0.035, 0.05, 0.08],
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__min_child_weight": [1, 2, 4, 7],
            "model__subsample": [0.7, 0.82, 0.92, 1.0],
            "model__colsample_bytree": [0.7, 0.82, 0.92, 1.0],
            "model__reg_alpha": [0.0, 0.03, 0.1, 0.3],
            "model__reg_lambda": [0.8, 1.2, 1.8, 3.0],
        }

    if model_name == "LightGBM":
        return {
            "model__n_estimators": [250, 350, 500, 700],
            "model__learning_rate": [0.02, 0.035, 0.05, 0.08],
            "model__num_leaves": [15, 23, 31, 47, 63],
            "model__min_child_samples": [10, 20, 30, 50],
            "model__subsample": [0.7, 0.82, 0.92, 1.0],
            "model__colsample_bytree": [0.7, 0.82, 0.92, 1.0],
            "model__reg_alpha": [0.0, 0.03, 0.1, 0.3],
            "model__reg_lambda": [0.8, 1.2, 1.8, 3.0],
        }

    if model_name == "Ridge Regression":
        return {"model__alpha": list(np.logspace(-3, 3, 20))}

    return {}


# ============================================================
# EVALUATION
# ============================================================

def evaluate_candidate(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    force_three=False,
) -> Dict[str, Any]:
    cv, folds = cv_strategy(y, problem_type, force_three=force_three)

    if problem_type == "classification":
        scoring = classification_scoring(y)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        row = {
            "model": name,
            "accuracy": safe_float(np.mean(scores["test_accuracy"])),
            "accuracy_std": safe_float(np.std(scores["test_accuracy"])),
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

        # Robust ranking rewards performance but lightly penalizes instability.
        auc_component = row["roc_auc"] if row["roc_auc"] is not None else row["f1"]
        row["ranking_score"] = safe_float(
            0.55 * row["f1"]
            + 0.25 * row["accuracy"]
            + 0.20 * auc_component
            - 0.08 * (row["f1_std"] or 0.0)
        )
        return row

    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    scores = cross_validate(
        pipeline,
        X,
        y,
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
    row["ranking_score"] = safe_float(row["r2"] - 0.08 * (row["r2_std"] or 0.0))
    return row


def tune_pipeline(
    model_name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
) -> Tuple[Pipeline, Dict[str, Any]]:
    params = tuning_space(model_name, problem_type)
    if not params:
        return pipeline, {"tuned": False}

    iterations = (
        TUNING_ITERATIONS_SMALL if len(X) <= SMALL_DATASET_ROWS
        else TUNING_ITERATIONS_LARGE
    )

    cv, folds = cv_strategy(y, problem_type, force_three=True)
    scoring = "f1_weighted" if problem_type == "classification" else "r2"

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=iterations,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)

    return search.best_estimator_, {
        "tuned": True,
        "iterations": iterations,
        "cv_folds": folds,
        "best_score": safe_float(search.best_score_),
        "best_params": {
            k.replace("model__", ""): v
            for k, v in search.best_params_.items()
        },
    }


def build_ensemble(
    ranked_candidates: List[Tuple[str, Pipeline]],
    problem_type: str,
):
    top = ranked_candidates[:3]
    if len(top) < 2:
        return None

    estimators = [
        (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"), clone(pipe))
        for name, pipe in top
    ]

    if problem_type == "classification":
        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            flatten_transform=True,
            n_jobs=1,
        )

    return VotingRegressor(
        estimators=estimators,
        n_jobs=1,
    )


def tune_binary_threshold(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> Optional[Dict[str, Any]]:
    if y.nunique() != 2:
        return None

    # Threshold tuning is intentionally limited for compute safety.
    if len(X) > SMALL_DATASET_ROWS:
        return None

    cv, folds = cv_strategy(y, "classification", force_three=True)

    try:
        prob = cross_val_predict(
            clone(estimator),
            X,
            y,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )
    except Exception:
        return None

    classes = np.sort(pd.unique(y))
    positive_class = classes[1]
    y_binary = (np.asarray(y) == positive_class).astype(int)

    best_threshold = 0.50
    best_score = -1.0

    for threshold in np.linspace(0.25, 0.75, 101):
        pred_binary = (prob[:, 1] >= threshold).astype(int)
        score = f1_score(y_binary, pred_binary, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return {
        "threshold": round(best_threshold, 3),
        "oof_positive_f1": safe_float(best_score),
        "positive_class": str(positive_class),
        "cv_folds": folds,
    }


def evaluate_holdout(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
    threshold_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    if problem_type == "classification":
        pred = estimator.predict(X_test)

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
            "threshold_used": 0.50,
        }

        try:
            prob = estimator.predict_proba(X_test)
            classes = estimator.classes_

            if len(classes) == 2:
                result["roc_auc"] = safe_float(
                    roc_auc_score(y_test, prob[:, 1])
                )

                # Apply OOF-tuned threshold only after tuning was completed
                # without touching the holdout set.
                if threshold_info:
                    threshold = float(threshold_info["threshold"])
                    positive_class = classes[1]
                    negative_class = classes[0]
                    tuned_pred = np.where(
                        prob[:, 1] >= threshold,
                        positive_class,
                        negative_class,
                    )

                    result.update({
                        "accuracy": safe_float(accuracy_score(y_test, tuned_pred)),
                        "precision": safe_float(
                            precision_score(
                                y_test, tuned_pred,
                                average="weighted",
                                zero_division=0,
                            )
                        ),
                        "recall": safe_float(
                            recall_score(
                                y_test, tuned_pred,
                                average="weighted",
                                zero_division=0,
                            )
                        ),
                        "f1": safe_float(
                            f1_score(
                                y_test, tuned_pred,
                                average="weighted",
                                zero_division=0,
                            )
                        ),
                        "threshold_used": threshold,
                    })

            elif len(classes) <= 20:
                result["roc_auc"] = safe_float(
                    roc_auc_score(
                        y_test,
                        prob,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
        except Exception:
            pass

        return result

    pred = estimator.predict(X_test)
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
    return {
        "status": "ok",
        "service": "john-automl-api",
        "version": "3.0.0",
        "xgboost": HAS_XGBOOST,
        "lightgbm": HAS_LIGHTGBM,
    }


@app.get("/health")
def health():
    return {"status": "healthy", "version": "3.0.0"}


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

    compressed_bytes = len(raw)

    if compressed == "gzip":
        try:
            raw = gzip.decompress(raw)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="The compressed CSV could not be decompressed.",
            )

    decompressed_bytes = len(raw)

    if decompressed_bytes > MAX_DECOMPRESSED_BYTES:
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
        raise HTTPException(
            status_code=400,
            detail=f'Target column "{target}" was not found.',
        )

    original_rows = len(df)
    original_columns = len(df.columns)

    # Remove exact duplicates transparently.
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = before_dedup - len(df)

    # Remove only rows whose outcome is unknown.
    missing_target_rows = int(df[target].isna().sum())
    df = df[df[target].notna()].copy()

    if len(df) < 30:
        raise HTTPException(
            status_code=400,
            detail="At least 30 rows with a non-missing target are required.",
        )

    df, memory_info = optimize_dataframe_dtypes(df)

    y = df[target].copy()
    X = df.drop(columns=[target]).copy()

    # Target-independent feature engineering.
    X, engineered_features = auto_feature_engineer(X)

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

    # Diagnose target leakage/proxy leakage before model selection.
    trust_report = leakage_diagnostics(X, y, target, problem_type)
    severe_leakage_features = trust_report.get("severe_exclusions", [])

    if severe_leakage_features:
        X = X.drop(columns=severe_leakage_features, errors="ignore")

    possible_group_column = detect_possible_group_column(X)
    possible_time_column = detect_possible_time_column(X)

    if X.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="All predictor columns were removed by severe leakage safeguards.",
        )

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

    # Untouched final evaluation set.
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

    X_select, y_select, sample_method = representative_sample(
        X_train_full,
        y_train_full,
        problem_type,
        MODEL_SELECTION_MAX_ROWS,
    )

    preprocessor = make_preprocessor(X_select)
    model_defs = build_models(problem_type)

    baseline_rows = []
    candidate_pipelines = {}

    # --------------------------------------------------------
    # Stage 1: broad model comparison
    # --------------------------------------------------------
    for name, estimator in model_defs.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ])

        try:
            row = evaluate_candidate(
                name,
                pipeline,
                X_select,
                y_select,
                problem_type,
            )
            baseline_rows.append(row)
            candidate_pipelines[name] = pipeline

        except Exception as exc:
            baseline_rows.append({
                "model": name,
                "error": f"{type(exc).__name__}: {str(exc)[:220]}",
            })

    valid_baselines = [
        r for r in baseline_rows
        if r.get("ranking_score") is not None
    ]

    if not valid_baselines:
        raise HTTPException(
            status_code=422,
            detail="None of the candidate models could be trained on this dataset.",
        )

    valid_baselines.sort(
        key=lambda r: r["ranking_score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # Stage 2: tune the strongest candidates
    # --------------------------------------------------------
    tuned_models = {}
    tuning_meta = {}

    top_n_to_tune = 2 if len(X_select) <= SMALL_DATASET_ROWS else 1

    for row in valid_baselines[:top_n_to_tune]:
        name = row["model"]
        pipe = candidate_pipelines[name]

        try:
            tuned_pipe, meta = tune_pipeline(
                name,
                pipe,
                X_select,
                y_select,
                problem_type,
            )

            tuned_eval = evaluate_candidate(
                name,
                tuned_pipe,
                X_select,
                y_select,
                problem_type,
                force_three=True,
            )
            tuned_eval["tuned"] = bool(meta.get("tuned"))

            # Keep tuning only when it actually improves the robust score.
            if (
                tuned_eval.get("ranking_score") is not None
                and tuned_eval["ranking_score"] > row["ranking_score"]
            ):
                tuned_models[name] = tuned_pipe
                tuning_meta[name] = meta

                # Replace the baseline row shown to the user.
                for i, existing in enumerate(baseline_rows):
                    if existing.get("model") == name:
                        baseline_rows[i] = tuned_eval
                        break
            else:
                tuned_models[name] = pipe
                tuning_meta[name] = {
                    **meta,
                    "accepted": False,
                    "reason": "Tuned configuration did not improve cross-validation ranking score.",
                }

        except Exception as exc:
            tuned_models[name] = pipe
            tuning_meta[name] = {
                "tuned": False,
                "error": f"{type(exc).__name__}: {str(exc)[:180]}",
            }

    # Re-rank after accepted tuning.
    valid_after_tuning = [
        r for r in baseline_rows
        if r.get("ranking_score") is not None
    ]
    valid_after_tuning.sort(
        key=lambda r: r["ranking_score"],
        reverse=True,
    )

    ranked_pipelines = []
    for row in valid_after_tuning:
        name = row["model"]
        pipe = tuned_models.get(name, candidate_pipelines.get(name))
        if pipe is not None:
            ranked_pipelines.append((name, pipe))

    # --------------------------------------------------------
    # Stage 3: optional soft-voting ensemble on smaller data
    # --------------------------------------------------------
    ensemble_used = False

    if len(X_select) <= SMALL_DATASET_ROWS:
        ensemble = build_ensemble(ranked_pipelines, problem_type)

        if ensemble is not None:
            try:
                ensemble_row = evaluate_candidate(
                    "Elite Ensemble",
                    ensemble,
                    X_select,
                    y_select,
                    problem_type,
                    force_three=True,
                )

                baseline_rows.append(ensemble_row)

                current_best = max(
                    valid_after_tuning,
                    key=lambda r: r["ranking_score"],
                )

                if ensemble_row["ranking_score"] > current_best["ranking_score"]:
                    ranked_pipelines.insert(0, ("Elite Ensemble", ensemble))
                    ensemble_used = True

            except Exception:
                pass

    # Final selection.
    final_valid_rows = [
        r for r in baseline_rows
        if r.get("ranking_score") is not None
    ]
    final_valid_rows.sort(
        key=lambda r: r["ranking_score"],
        reverse=True,
    )

    best_row = final_valid_rows[0]
    best_model_name = best_row["model"]

    if best_model_name == "Elite Ensemble":
        best_estimator = build_ensemble(ranked_pipelines[1:4], problem_type)
        ensemble_used = True
    else:
        best_estimator = (
            tuned_models.get(best_model_name)
            or candidate_pipelines[best_model_name]
        )

    selection_metric = (
        "Robust classification score (F1 + accuracy + ROC-AUC + stability)"
        if problem_type == "classification"
        else "R² with stability penalty"
    )

    # --------------------------------------------------------
    # Stage 4: optional OOF threshold optimization
    # --------------------------------------------------------
    threshold_info = None
    if problem_type == "classification":
        threshold_info = tune_binary_threshold(
            best_estimator,
            X_select,
            y_select,
        )

    # --------------------------------------------------------
    # Stage 5: retrain winner on full training split
    # --------------------------------------------------------
    final_estimator = clone(best_estimator)
    final_estimator.fit(X_train_full, y_train_full)

    holdout_metrics = evaluate_holdout(
        final_estimator,
        X_test,
        y_test,
        problem_type,
        threshold_info=threshold_info,
    )

    # Trust score is deliberately separate from model accuracy.
    # High predictive performance with leakage warnings should not be presented
    # as high confidence.
    high_risk = trust_report.get("high_risk_count", 0)
    medium_risk = trust_report.get("medium_risk_count", 0)

    trust_score = 100
    trust_score -= high_risk * 25
    trust_score -= medium_risk * 8

    if possible_group_column:
        trust_score -= 12
    if possible_time_column:
        trust_score -= 8

    trust_score = int(max(0, min(100, trust_score)))

    if trust_score >= 85:
        trust_label = "High"
    elif trust_score >= 65:
        trust_label = "Moderate"
    else:
        trust_label = "Low"

    elapsed = round(time.time() - started, 2)

    compression_ratio = (
        round(decompressed_bytes / compressed_bytes, 2)
        if compressed == "gzip" and compressed_bytes > 0
        else 1.0
    )

    return {
        "api_version": "3.0.0",
        "problem_type": problem_type,
        "target": target,
        "best_model": best_model_name,
        "selection_metric": selection_metric,
        "models": baseline_rows,
        "rows_used": int(len(df)),
        "features_used": int(X.shape[1]),
        "original_features": int(original_columns - 1),
        "holdout_rows": int(len(X_test)),
        "model_selection_rows": int(len(X_select)),
        "final_training_rows": int(len(X_train_full)),
        "final_test_metrics": holdout_metrics,
        "threshold_optimization": threshold_info,
        "tuning": tuning_meta,
        "ensemble_used": ensemble_used,
        "available_engines": {
            "xgboost": HAS_XGBOOST,
            "lightgbm": HAS_LIGHTGBM,
        },
        "trust": {
            "score": trust_score,
            "label": trust_label,
            "leakage_findings": trust_report.get("findings", []),
            "automatically_excluded_leakage_features": severe_leakage_features,
            "possible_group_column": possible_group_column,
            "possible_time_column": possible_time_column,
            "warning": (
                "Model quality and trust are different. High accuracy can still be misleading "
                "when target proxies, repeated entities, or temporal leakage exist."
            ),
        },
        "optimization": {
            **memory_info,
            "original_rows": int(original_rows),
            "original_columns": int(original_columns),
            "duplicate_rows_removed": int(duplicate_rows_removed),
            "missing_target_rows_removed": int(missing_target_rows),
            "sampling_applied": bool(len(X_select) < len(X_train_full)),
            "sampling_method": sample_method,
            "engineered_features": engineered_features,
            "engineered_feature_count": len(engineered_features),
            "dropped_constant_features": constants,
            "dropped_identifier_features": identifier_like,
            "dropped_free_text_features": free_text,
        },
        "processing_seconds": elapsed,
        "upload_compression": "gzip" if compressed == "gzip" else "none",
        "compression_ratio": compression_ratio,
        "compressed_upload_mb": round(compressed_bytes / 1024 / 1024, 2),
        "original_upload_mb": round(decompressed_bytes / 1024 / 1024, 2),
        "source_filename": original_name or file.filename,
    }
