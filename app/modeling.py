import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROBLEM TYPE DETECTION
# ============================================================

def detect_problem_type(y):
    unique_count = y.nunique(dropna=True)
    unique_ratio = unique_count / max(len(y), 1)

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    if unique_count <= 20 or unique_ratio <= 0.05:
        return "classification"

    return "regression"


# ============================================================
# FEATURE CLEANUP
# ============================================================

def remove_problematic_features(X):
    """
    Removes columns that are very likely to be identifiers or
    extremely high-cardinality text fields.

    These columns can create enormous one-hot encoded matrices.
    """

    drop_columns = []

    for column in X.columns:
        series = X[column]
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / max(len(series), 1)

        # Very likely ID / row identifier
        if unique_ratio >= 0.95:
            drop_columns.append(column)
            continue

        # Very high-cardinality categorical feature
        if (
            not pd.api.types.is_numeric_dtype(series)
            and unique_count > 100
            and unique_ratio > 0.50
        ):
            drop_columns.append(column)

    cleaned = X.drop(columns=drop_columns, errors="ignore")

    return cleaned, drop_columns


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X):
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def train_classification_models(X, y):
    class_counts = y.value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Classification requires at least two target classes."
        )

    minimum_class_count = int(class_counts.min())

    if minimum_class_count < 2:
        raise ValueError(
            "At least two examples are required for every target class."
        )

    folds = min(5, minimum_class_count)

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    binary_problem = y.nunique() == 2

    if binary_problem:
        scoring["roc_auc"] = "roc_auc"
    else:
        scoring["roc_auc"] = "roc_auc_ovr_weighted"

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("model", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        results.append(
            {
                "model": model_name,
                "accuracy": round(
                    float(np.mean(scores["test_accuracy"])), 4
                ),
                "accuracy_std": round(
                    float(np.std(scores["test_accuracy"])), 4
                ),
                "precision": round(
                    float(np.mean(scores["test_precision"])), 4
                ),
                "recall": round(
                    float(np.mean(scores["test_recall"])), 4
                ),
                "f1": round(
                    float(np.mean(scores["test_f1"])), 4
                ),
                "f1_std": round(
                    float(np.std(scores["test_f1"])), 4
                ),
                "roc_auc": round(
                    float(np.mean(scores["test_roc_auc"])), 4
                ),
                "cv_folds": folds,
            }
        )

    results.sort(
        key=lambda item: item["f1"],
        reverse=True,
    )

    return results


# ============================================================
# REGRESSION
# ============================================================

def train_regression_models(X, y):
    folds = min(5, len(y))

    if folds < 2:
        raise ValueError(
            "Not enough rows are available for cross-validation."
        )

    cv = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("model", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        results.append(
            {
                "model": model_name,
                "r2": round(
                    float(np.mean(scores["test_r2"])), 4
                ),
                "r2_std": round(
                    float(np.std(scores["test_r2"])), 4
                ),
                "mae": round(
                    float(-np.mean(scores["test_mae"])), 4
                ),
                "rmse": round(
                    float(-np.mean(scores["test_rmse"])), 4
                ),
                "cv_folds": folds,
            }
        )

    results.sort(
        key=lambda item: item["r2"],
        reverse=True,
    )

    return results


# ============================================================
# MAIN MODELING FUNCTION
# ============================================================

def run_modeling(dataframe, target):
    if target not in dataframe.columns:
        raise ValueError(
            f"Target column '{target}' was not found in the dataset."
        )

    working_data = dataframe.dropna(subset=[target]).copy()

    if len(working_data) < 20:
        raise ValueError(
            "At least 20 rows with a non-missing target are required."
        )

    y = working_data[target]

    X = working_data.drop(
        columns=[target]
    )

    X, dropped_features = remove_problematic_features(X)

    if X.shape[1] == 0:
        raise ValueError(
            "No usable predictor columns remain after preprocessing."
        )

    problem_type = detect_problem_type(y)

    if problem_type == "classification":
        models = train_classification_models(X, y)
        selection_metric = "F1 score"
    else:
        models = train_regression_models(X, y)
        selection_metric = "R²"

    return {
        "problem_type": problem_type,
        "target": target,
        "rows_used": int(len(working_data)),
        "features_used": int(X.shape[1]),
        "dropped_features": dropped_features,
        "selection_metric": selection_metric,
        "best_model": models[0]["model"],
        "models": models,
    }
