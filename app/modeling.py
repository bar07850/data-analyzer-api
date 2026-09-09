import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_validate,
    RandomizedSearchCV,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


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
    drop_columns = []

    for column in X.columns:
        series = X[column]

        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / max(len(series), 1)
        missing_ratio = series.isna().mean()

        # Completely empty / almost completely empty
        if missing_ratio >= 0.98:
            drop_columns.append(column)
            continue

        # Constant feature
        if unique_count <= 1:
            drop_columns.append(column)
            continue

        # Likely identifier
        if unique_ratio >= 0.95:
            drop_columns.append(column)
            continue

        # Very high-cardinality categorical column
        if (
            not pd.api.types.is_numeric_dtype(series)
            and unique_count > 100
            and unique_ratio > 0.50
        ):
            drop_columns.append(column)

    cleaned = X.drop(
        columns=drop_columns,
        errors="ignore",
    )

    return cleaned, drop_columns


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X):
    numeric_columns = (
        X.select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
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
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )


# ============================================================
# CLASSIFICATION HELPERS
# ============================================================

def classification_scoring(y):
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    if y.nunique() == 2:
        scoring["roc_auc"] = "roc_auc"
    else:
        scoring["roc_auc"] = "roc_auc_ovr_weighted"

    return scoring


def detect_class_imbalance(y):
    counts = y.value_counts()

    if len(counts) < 2:
        return False

    largest = counts.max()
    smallest = counts.min()

    if smallest == 0:
        return True

    return (largest / smallest) >= 2.0


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
        random_state=RANDOM_STATE,
    )

    imbalanced = detect_class_imbalance(y)

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "model__C": [
                    0.01,
                    0.1,
                    0.5,
                    1.0,
                    2.0,
                    10.0,
                    100.0,
                ],
                "model__class_weight": (
                    [None, "balanced"]
                    if imbalanced
                    else [None]
                ),
            },
        },

        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "model__n_estimators": [
                    100,
                    200,
                    300,
                    500,
                ],
                "model__max_depth": [
                    None,
                    5,
                    10,
                    20,
                    30,
                ],
                "model__min_samples_split": [
                    2,
                    5,
                    10,
                ],
                "model__min_samples_leaf": [
                    1,
                    2,
                    4,
                ],
                "model__max_features": [
                    "sqrt",
                    "log2",
                    None,
                ],
                "model__class_weight": (
                    [None, "balanced"]
                    if imbalanced
                    else [None]
                ),
            },
        },
    }

    scoring = classification_scoring(y)

    results = []

    for model_name, config in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(X),
                ),
                (
                    "model",
                    config["model"],
                ),
            ]
        )

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        baseline_scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        baseline_f1 = float(
            np.mean(
                baseline_scores["test_f1"]
            )
        )

        # ----------------------------------------------------
        # HYPERPARAMETER OPTIMIZATION
        # ----------------------------------------------------

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=config["params"],
            n_iter=15,
            scoring="f1_weighted",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        search.fit(X, y)

        best_pipeline = search.best_estimator_

        # ----------------------------------------------------
        # EVALUATE TUNED MODEL
        # ----------------------------------------------------

        tuned_scores = cross_validate(
            best_pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        tuned_f1 = float(
            np.mean(
                tuned_scores["test_f1"]
            )
        )

        improvement = (
            tuned_f1 - baseline_f1
        )

        clean_parameters = {
            key.replace("model__", ""): value
            for key, value
            in search.best_params_.items()
        }

        results.append(
            {
                "model": model_name,

                "accuracy": round(
                    float(
                        np.mean(
                            tuned_scores[
                                "test_accuracy"
                            ]
                        )
                    ),
                    4,
                ),

                "accuracy_std": round(
                    float(
                        np.std(
                            tuned_scores[
                                "test_accuracy"
                            ]
                        )
                    ),
                    4,
                ),

                "precision": round(
                    float(
                        np.mean(
                            tuned_scores[
                                "test_precision"
                            ]
                        )
                    ),
                    4,
                ),

                "recall": round(
                    float(
                        np.mean(
                            tuned_scores[
                                "test_recall"
                            ]
                        )
                    ),
                    4,
                ),

                "f1": round(
                    tuned_f1,
                    4,
                ),

                "f1_std": round(
                    float(
                        np.std(
                            tuned_scores[
                                "test_f1"
                            ]
                        )
                    ),
                    4,
                ),

                "roc_auc": round(
                    float(
                        np.mean(
                            tuned_scores[
                                "test_roc_auc"
                            ]
                        )
                    ),
                    4,
                ),

                "baseline_f1": round(
                    baseline_f1,
                    4,
                ),

                "improvement": round(
                    improvement,
                    4,
                ),

                "best_params": clean_parameters,

                "cv_folds": folds,
            }
        )

    results.sort(
        key=lambda item: item["f1"],
        reverse=True,
    )

    return results, imbalanced


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
        random_state=RANDOM_STATE,
    )

    models = {

        "Linear Regression": {
            "model": LinearRegression(),
            "params": {},
        },

        "Ridge Regression": {
            "model": Ridge(),
            "params": {
                "model__alpha": [
                    0.001,
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                    1000.0,
                ],
            },
        },

        "Random Forest Regressor": {
            "model": RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "model__n_estimators": [
                    100,
                    200,
                    300,
                    500,
                ],
                "model__max_depth": [
                    None,
                    5,
                    10,
                    20,
                    30,
                ],
                "model__min_samples_split": [
                    2,
                    5,
                    10,
                ],
                "model__min_samples_leaf": [
                    1,
                    2,
                    4,
                ],
                "model__max_features": [
                    1.0,
                    "sqrt",
                    "log2",
                ],
            },
        },
    }

    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }

    results = []

    for model_name, config in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(X),
                ),
                (
                    "model",
                    config["model"],
                ),
            ]
        )

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        baseline_scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        baseline_r2 = float(
            np.mean(
                baseline_scores["test_r2"]
            )
        )

        # Linear Regression has no meaningful tuning here.
        if not config["params"]:

            best_pipeline = pipeline
            best_params = {}

        else:

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=config["params"],
                n_iter=min(
                    15,
                    np.prod(
                        [
                            len(values)
                            for values
                            in config["params"].values()
                        ]
                    ),
                ),
                scoring="r2",
                cv=cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )

            search.fit(X, y)

            best_pipeline = search.best_estimator_

            best_params = {
                key.replace("model__", ""): value
                for key, value
                in search.best_params_.items()
            }

        # ----------------------------------------------------
        # EVALUATE TUNED MODEL
        # ----------------------------------------------------

        tuned_scores = cross_validate(
            best_pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        tuned_r2 = float(
            np.mean(
                tuned_scores["test_r2"]
            )
        )

        improvement = (
            tuned_r2 - baseline_r2
        )

        results.append(
            {
                "model": model_name,

                "r2": round(
                    tuned_r2,
                    4,
                ),

                "r2_std": round(
                    float(
                        np.std(
                            tuned_scores[
                                "test_r2"
                            ]
                        )
                    ),
                    4,
                ),

                "mae": round(
                    float(
                        -np.mean(
                            tuned_scores[
                                "test_mae"
                            ]
                        )
                    ),
                    4,
                ),

                "rmse": round(
                    float(
                        -np.mean(
                            tuned_scores[
                                "test_rmse"
                            ]
                        )
                    ),
                    4,
                ),

                "baseline_r2": round(
                    baseline_r2,
                    4,
                ),

                "improvement": round(
                    improvement,
                    4,
                ),

                "best_params": best_params,

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

    working_data = (
        dataframe
        .dropna(subset=[target])
        .copy()
    )

    if len(working_data) < 20:
        raise ValueError(
            "At least 20 rows with a non-missing target are required."
        )

    y = working_data[target]

    X = working_data.drop(
        columns=[target]
    )

    X, dropped_features = (
        remove_problematic_features(X)
    )

    if X.shape[1] == 0:
        raise ValueError(
            "No usable predictor columns remain after preprocessing."
        )

    problem_type = detect_problem_type(y)

    class_imbalance = False

    if problem_type == "classification":

        models, class_imbalance = (
            train_classification_models(
                X,
                y,
            )
        )

        selection_metric = "F1 score"

    else:

        models = train_regression_models(
            X,
            y,
        )

        selection_metric = "R²"

    return {
        "problem_type": problem_type,
        "target": target,
        "rows_used": int(
            len(working_data)
        ),
        "features_used": int(
            X.shape[1]
        ),
        "dropped_features": dropped_features,
        "selection_metric": selection_metric,
        "class_imbalance_detected": (
            class_imbalance
        ),
        "optimization": (
            "RandomizedSearchCV"
        ),
        "best_model": (
            models[0]["model"]
        ),
        "models": models,
    }
