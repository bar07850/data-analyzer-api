import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)

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


def detect_problem_type(y: pd.Series) -> str:
    """
    Determine whether the target is more likely
    classification or regression.
    """

    y_clean = y.dropna()

    if y_clean.empty:
        raise ValueError("Target column contains no usable values.")

    if not pd.api.types.is_numeric_dtype(y_clean):
        return "classification"

    unique_count = y_clean.nunique()
    unique_ratio = unique_count / len(y_clean)

    if unique_count <= 20 or unique_ratio <= 0.05:
        return "classification"

    return "regression"


def build_preprocessor(X: pd.DataFrame):

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()


    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )


    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ]
    )

    return preprocessor


def train_classification_models(
    X: pd.DataFrame,
    y: pd.Series
):

    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=2000
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            ),
    }


    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )


    results = []


    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                ),
            ]
        )


        pipeline.fit(
            X_train,
            y_train
        )


        predictions = pipeline.predict(
            X_test
        )


        accuracy = accuracy_score(
            y_test,
            predictions
        )


        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )


        result = {
            "model": name,
            "accuracy": round(
                float(accuracy),
                4
            ),
            "precision": round(
                float(precision),
                4
            ),
            "recall": round(
                float(recall),
                4
            ),
            "f1": round(
                float(f1),
                4
            ),
        }


        if y.nunique() == 2:

            try:

                probabilities = (
                    pipeline.predict_proba(
                        X_test
                    )[:, 1]
                )

                auc = roc_auc_score(
                    y_test,
                    probabilities
                )

                result["roc_auc"] = round(
                    float(auc),
                    4
                )

            except Exception:

                result["roc_auc"] = None


        results.append(
            result
        )


    results.sort(
        key=lambda x: x["f1"],
        reverse=True
    )


    return results


def train_regression_models(
    X: pd.DataFrame,
    y: pd.Series
):

    preprocessor = build_preprocessor(X)


    models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            ),
    }


    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )


    results = []


    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                ),
            ]
        )


        pipeline.fit(
            X_train,
            y_train
        )


        predictions = pipeline.predict(
            X_test
        )


        mae = mean_absolute_error(
            y_test,
            predictions
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )


        r2 = r2_score(
            y_test,
            predictions
        )


        results.append(
            {
                "model": name,
                "mae": round(
                    float(mae),
                    4
                ),
                "rmse": round(
                    float(rmse),
                    4
                ),
                "r2": round(
                    float(r2),
                    4
                ),
            }
        )


    results.sort(
        key=lambda x: x["r2"],
        reverse=True
    )


    return results


def run_modeling(
    dataframe: pd.DataFrame,
    target: str
):

    if target not in dataframe.columns:
        raise ValueError(
            f"Target column '{target}' was not found."
        )


    df = dataframe.copy()


    df = df.dropna(
        subset=[target]
    )


    if len(df) < 20:
        raise ValueError(
            "Dataset needs at least 20 usable rows for modeling."
        )


    y = df[target]

    X = df.drop(
        columns=[target]
    )


    if X.shape[1] == 0:
        raise ValueError(
            "No predictor columns remain after selecting the target."
        )


    problem_type = detect_problem_type(
        y
    )


    if problem_type == "classification":

        model_results = (
            train_classification_models(
                X,
                y
            )
        )

    else:

        model_results = (
            train_regression_models(
                X,
                y
            )
        )


    return {
        "problem_type":
            problem_type,

        "target":
            target,

        "rows_used":
            len(df),

        "features_used":
            X.shape[1],

        "best_model":
            model_results[0][
                "model"
            ],

        "models":
            model_results,
    }
