
"""Automatic machine-learning model selection and evaluation."""

from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _json_value(value: Any) -> Any:
    """Convert common NumPy/Pandas values to JSON-safe values."""
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create separate preprocessing pipelines for numeric and text columns."""
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError("No usable input features were found.")

    return ColumnTransformer(transformers=transformers)


def _candidate_models(task: str) -> dict[str, Any]:
    """Return candidate models appropriate for the selected task."""
    if task == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            ),
        }

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }


def _regression_metrics(actual, predicted) -> dict[str, Any]:
    """Calculate standard regression metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": (
            float(r2_score(actual, predicted))
            if len(actual) >= 2
            else None
        ),
    }


def _classification_metrics(actual, predicted) -> dict[str, Any]:
    """Calculate weighted classification metrics."""
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "accuracy_percent": round(
            float(accuracy_score(actual, predicted)) * 100, 2
        ),
        "precision_weighted": float(
            precision_score(
                actual, predicted,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                actual, predicted,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                actual, predicted,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def _feature_importances(model, feature_names) -> list[dict[str, Any]]:
    """Return feature importance or coefficient magnitude when supported."""
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessing"]

    try:
        names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        names = feature_names

    values = getattr(estimator, "feature_importances_", None)

    if values is None and hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        values = np.mean(np.abs(coefficients), axis=0)

    if values is None:
        return []

    values = np.asarray(values).ravel()

    items = [
        {
            "feature": str(names[index]) if index < len(names) else f"feature_{index}",
            "importance": float(value),
        }
        for index, value in enumerate(values)
        if np.isfinite(value)
    ]

    items.sort(key=lambda item: item["importance"], reverse=True)
    return items[:20]


def train_and_evaluate(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "auto",
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Select a model on validation data and evaluate it on held-out test data."""

    warnings: list[str] = []

    if df.empty or len(df.columns) < 2:
        raise ValueError(
            "Provide a dataset with a target and at least one feature column."
        )

    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")

    if task_type not in {"auto", "regression", "classification"}:
        raise ValueError(
            "task_type must be auto, regression, or classification."
        )

    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")

    data = df.copy()
    data.columns = data.columns.map(str)
    target_column = str(target_column)

    if target_column not in data.columns:
        raise ValueError(
            f"Target '{target_column}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    data = data.replace([np.inf, -np.inf], np.nan)

    # Convert dates to text features. Date-specific extraction can be added
    # later as a separate feature-engineering operation.
    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = data[column].map(
                lambda value: (
                    value.isoformat() if pd.notna(value) else np.nan
                )
            )

    data = data.dropna(subset=[target_column]).copy()

    if len(data) < 30:
        raise ValueError(
            "At least 30 rows with known target values are required "
            "for model comparison. More data is recommended."
        )

    y = data[target_column]
    X = data.drop(columns=[target_column])

    # Remove features that contain no useful variation.
    constant_columns = [
        column for column in X.columns
        if X[column].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        X = X.drop(columns=constant_columns)
        warnings.append(
            "Constant feature columns were removed: "
            + ", ".join(map(str, constant_columns))
        )

    if X.shape[1] == 0:
        raise ValueError("No varying feature columns remain.")

    if y.nunique(dropna=True) < 2:
        raise ValueError("The target must contain at least two distinct values.")

    # Automatic detection is a heuristic. Users can override it in the API.
    if task_type == "auto":
        if (
            pd.api.types.is_numeric_dtype(y)
            and y.nunique() > 15
        ):
            selected_task = "regression"
        else:
            selected_task = "classification"
    else:
        selected_task = task_type

    if selected_task == "regression":
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid].astype(float)

        if len(y) < 30 or y.nunique() < 2:
            raise ValueError(
                "Insufficient valid numeric target values for regression."
            )
    else:
        y = y.astype(str)
        counts = y.value_counts()

        if len(counts) < 2:
            raise ValueError("Classification requires at least two classes.")

        if counts.min() < 4:
            raise ValueError(
                "Each class needs at least 4 rows for this evaluation. "
                "Collect more examples or choose a different target."
            )

    # Reserve the test set before selecting models. It is not used for
    # model selection.
    stratify_test = None
    if selected_task == "classification":
        counts = y.value_counts()
        test_rows = int(np.ceil(len(y) * test_size))
        train_rows = len(y) - test_rows

        if counts.min() >= 2 and test_rows >= len(counts) and train_rows >= len(counts):
            stratify_test = y
        else:
            warnings.append(
                "The class distribution is too small for a fully stratified "
                "test split; metrics may be less reliable."
            )

    try:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_test,
        )
    except ValueError as exc:
        raise ValueError(f"Could not split the dataset: {exc}") from exc

    # Create a validation split from the training partition only.
    stratify_validation = None
    if selected_task == "classification":
        counts = y_train_full.value_counts()
        validation_rows = max(1, int(np.ceil(len(y_train_full) * 0.2)))
        training_rows = len(y_train_full) - validation_rows

        if counts.min() >= 2 and validation_rows >= len(counts) and training_rows >= len(counts):
            stratify_validation = y_train_full
        else:
            warnings.append(
                "Validation data cannot include every class reliably. "
                "Model selection may be unstable."
            )

    try:
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=stratify_validation,
        )
    except ValueError as exc:
        raise ValueError(
            f"Could not create a validation split: {exc}"
        ) from exc

    if y_train.nunique() < 2:
        raise ValueError("The training split contains only one target value/class.")

    comparison = []
    fitted_candidates = {}

    for model_name, estimator in _candidate_models(selected_task).items():
        try:
            pipeline = Pipeline(
                steps=[
                    ("preprocessing", _make_preprocessor(X_train)),
                    ("model", estimator),
                ]
            )

            pipeline.fit(X_train, y_train)
            validation_predictions = pipeline.predict(X_validation)

            if selected_task == "regression":
                metrics = _regression_metrics(
                    y_validation,
                    validation_predictions,
                )
                # Minimize validation RMSE.
                selection_score = metrics["rmse"]
                comparison_score = selection_score
                metric_name = "rmse"
            else:
                metrics = _classification_metrics(
                    y_validation,
                    validation_predictions,
                )
                # Maximize validation weighted F1.
                selection_score = metrics["f1_weighted"]
                comparison_score = selection_score
                metric_name = "f1_weighted"

            comparison.append(
                {
                    "model": model_name,
                    "validation_metrics": metrics,
                    "selection_metric": metric_name,
                    "selection_score": float(comparison_score),
                    "status": "success",
                }
            )
            fitted_candidates[model_name] = estimator

        except (ValueError, TypeError, ArithmeticError) as exc:
            warnings.append(
                f"{model_name} could not be evaluated: {str(exc)[:250]}"
            )
            comparison.append(
                {
                    "model": model_name,
                    "validation_metrics": {},
                    "selection_metric": (
                        "rmse" if selected_task == "regression"
                        else "f1_weighted"
                    ),
                    "selection_score": None,
                    "status": "failed",
                    "error": str(exc)[:250],
                }
            )

    successful = [
        item for item in comparison
        if item["status"] == "success"
    ]

    if not successful:
        raise ValueError(
            "All candidate models failed. Check your feature values, "
            "target column, and dataset size."
        )

    if selected_task == "regression":
        winner = min(
            successful,
            key=lambda item: item["selection_score"],
        )
    else:
        winner = max(
            successful,
            key=lambda item: item["selection_score"],
        )

    best_name = winner["model"]

    # Refit the selected model on all training data, then test once.
    final_model = Pipeline(
        steps=[
            ("preprocessing", _make_preprocessor(X_train_full)),
            ("model", fitted_candidates[best_name]),
        ]
    )

    try:
        final_model.fit(X_train_full, y_train_full)
        predictions = final_model.predict(X_test)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"The selected model could not be trained: {exc}"
        ) from exc

    if selected_task == "regression":
        metrics = _regression_metrics(y_test, predictions)

        if metrics["r2"] is not None and metrics["r2"] < 0:
            warnings.append(
                "Test R² is below zero. This model performed worse than "
                "predicting the test-set mean on this test sample."
            )

        residuals = np.asarray(y_test, dtype=float) - np.asarray(
            predictions, dtype=float
        )

        prediction_chart = [
            {
                "actual": float(actual),
                "predicted": float(predicted),
                "residual": float(actual - predicted),
            }
            for actual, predicted in zip(
                y_test.iloc[:200].to_numpy(),
                np.asarray(predictions)[:200],
            )
        ]

        visualization = {
            "type": "regression",
            "actual_vs_predicted": prediction_chart,
            "residuals": [
                {
                    "predicted": float(predicted),
                    "residual": float(residual),
                }
                for predicted, residual in zip(
                    np.asarray(predictions)[:200],
                    residuals[:200],
                )
            ],
        }

    else:
        metrics = _classification_metrics(y_test, predictions)

        classes = sorted(
            set(map(str, y_test.unique()))
            | set(map(str, np.unique(predictions)))
        )

        matrix = confusion_matrix(
            y_test.astype(str),
            np.asarray(predictions).astype(str),
            labels=classes,
        )

        visualization = {
            "type": "classification",
            "confusion_matrix": {
                "labels": classes,
                "matrix": matrix.tolist(),
            },
            "class_counts": [
                {
                    "class": label,
                    "actual_count": int(
                        (y_test.astype(str) == label).sum()
                    ),
                    "predicted_count": int(
                        (np.asarray(predictions).astype(str) == label).sum()
                    ),
                }
                for label in classes
            ],
        }

    preview_count = min(100, len(y_test))
    preview = [
        {
            "actual": _json_value(actual),
            "predicted": _json_value(predicted),
        }
        for actual, predicted in zip(
            y_test.iloc[:preview_count].tolist(),
            list(predictions[:preview_count]),
        )
    ]

    if len(data) < 100:
        warnings.append(
            "The dataset is relatively small. Results may vary with other "
            "samples; evaluate on more data before deployment."
        )

    if selected_task == "classification" and len(
        set(map(str, y_test.unique()))
    ) < len(set(map(str, y.unique()))):
        warnings.append(
            "Not every target class appears in the test data. "
            "Review class coverage before relying on these metrics."
        )

    importance = _feature_importances(final_model, X.columns)

    return {
        "task_type": selected_task,
        "target_column": target_column,
        "selected_model": best_name,
        "model": best_name,
        "selection_metric": winner["selection_metric"],
        "selection_score": winner["selection_score"],
        "rows_used": int(len(data)),
        "training_rows": int(len(X_train_full)),
        "testing_rows": int(len(X_test)),
        "feature_columns": list(map(str, X.columns)),
        "removed_constant_columns": list(map(str, constant_columns)),
        "metrics": metrics,
        "model_comparison": comparison,
        "test_prediction_preview": preview,
        "visualization": visualization,
        "feature_importance": importance,
        "warnings": warnings,
        "evaluation_note": (
            "Models were selected using a validation split from the training "
            "data. Reported metrics were calculated on a separate held-out "
            "test set. Results estimate performance and do not guarantee "
            "future accuracy. This endpoint does not persist the trained model."
        ),
    }
