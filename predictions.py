"python"
"""
Automated Machine Learning module for Data Analyst Agent.

Features:
- Prediction target recommendations
- Automatic classification/regression detection
- Numeric and categorical preprocessing
- Missing-value imputation
- Model comparison using validation data
- Classification accuracy and weighted F1
- Regression MAE, RMSE, and R²
- Confusion matrix and classification report
- Feature importance
- Prediction previews
- Optional return of the fitted model for persistence

Important:
Evaluation metrics are estimates, not guarantees of future performance.
"""

# ============================================================
# 1. STANDARD LIBRARY IMPORTS
# ============================================================

import math
import re
from typing import Any


# ============================================================
# 2. THIRD-PARTY DATA PROCESSING IMPORTS
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# 3. SCIKIT-LEARN MODEL AND PIPELINE IMPORTS
# ============================================================

from sklearn.base import clone
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import train_test_split

from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor,
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ============================================================
# 4. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# Neural networks are considered only for datasets within this limit.
MAX_ROWS_FOR_NEURAL_NETWORK = 20_000

# Features with more unique categories than this are excluded
# from the current categorical preprocessing workflow.
MAX_CATEGORIES = 100

# Maximum number of classes supported by this workflow.
MAX_TARGET_CLASSES = 100


# ============================================================
# 5. JSON-SAFE VALUE CONVERSION
# ============================================================

def _clean_json(value: Any) -> Any:
    """Convert NumPy and Pandas values into JSON-safe Python values."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): _clean_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _clean_json(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return (
            float(value)
            if np.isfinite(value)
            else None
        )

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        return (
            value
            if math.isfinite(value)
            else None
        )

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if value is pd.NA or value is pd.NaT:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (str, int, bool)):
        return value

    return str(value)


# ============================================================
# 6. AUTOMATIC TASK DETECTION
# ============================================================

def _target_task(series: pd.Series) -> str:
    """
    Infer whether a target is suitable for classification or regression.

    Numeric targets with few repeated values may be classification targets.
    Other numeric targets are treated as regression targets.
    """

    non_null = series.dropna()

    if non_null.empty:
        raise ValueError(
            "The selected target column contains no usable values."
        )

    if (
        pd.api.types.is_bool_dtype(non_null)
        or pd.api.types.is_object_dtype(non_null)
        or isinstance(non_null.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(non_null)
    ):
        return "classification"

    unique_count = int(non_null.nunique())

    unique_ratio = (
        unique_count / max(len(non_null), 1)
    )

    if (
        unique_count <= 20
        and unique_ratio < 0.20
    ):
        return "classification"

    return "regression"


# ============================================================
# 7. TARGET COLUMN RECOMMENDATIONS
# ============================================================

def recommend_target_columns(
    df: pd.DataFrame,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Rank possible target columns using transparent heuristics.

    The score is a recommendation, not proof that a column is the
    correct target for the user's business objective.
    """

    if df.empty or len(df.columns) < 2:
        return {
            "can_train": False,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "recommendations": [],
            "message": (
                "Provide a dataset with rows and at least two columns."
            ),
        }

    recommendations: list[dict[str, Any]] = []

    row_count = len(df)

    target_words = re.compile(
        r"(target|label|outcome|class|status|churn|default|"
        r"survived|approved|diagnosis|price|sales|revenue|profit|"
        r"salary|score|rating|demand|quantity|total|amount|"
        r"response|converted|conversion|result|y$)",
        re.IGNORECASE,
    )

    identifier_words = re.compile(
        r"(^id$|_id$|^id_|uuid|guid|index|row_number|"
        r"customer.?number|account.?number)",
        re.IGNORECASE,
    )

    for column in df.columns:
        series = df[column]

        name = str(column)

        non_null = series.dropna()

        unique_count = int(
            non_null.nunique()
        )

        missing_ratio = float(
            series.isna().mean()
        )

        unique_ratio = (
            unique_count / max(len(non_null), 1)
        )

        reasons: list[str] = []

        score = 0.0

        if len(non_null) < 20:
            score -= 25

            reasons.append(
                "Few non-missing observations."
            )
        else:
            score += 10

            reasons.append(
                "Enough observed values for initial consideration."
            )

        if unique_count <= 1:
            score -= 100

            reasons.append(
                "Constant or empty column; unsuitable as a target."
            )

        if missing_ratio > 0.50:
            score -= 30

            reasons.append(
                "More than half the values are missing."
            )

        elif missing_ratio <= 0.05:
            score += 10

            reasons.append(
                "Low proportion of missing values."
            )

        if target_words.search(name):
            score += 25

            reasons.append(
                "Column name suggests an outcome or business measure."
            )

        if identifier_words.search(name):
            score -= 40

            reasons.append(
                "Column name suggests an identifier."
            )

        if (
            unique_ratio > 0.98
            and unique_count > 20
        ):
            score -= 30

            reasons.append(
                "Almost every value is unique."
            )

        if pd.api.types.is_datetime64_any_dtype(series):
            score -= 15

            reasons.append(
                "Date/time target may require a separate forecasting workflow."
            )

        elif pd.api.types.is_numeric_dtype(series):
            score += 15

            reasons.append(
                "Numeric target may support regression."
            )

        else:
            score += 10

            reasons.append(
                "Categorical target may support classification."
            )

        if (
            not pd.api.types.is_numeric_dtype(series)
            and unique_count > MAX_CATEGORIES
        ):
            score -= 15

            reasons.append(
                "High number of categories."
            )

        if row_count < 30:
            score -= 20

            reasons.append(
                "Small dataset; model evaluation may be unreliable."
            )

        suitable = (
            unique_count > 1
            and len(non_null) >= 20
            and missing_ratio <= 0.80
        )

        try:
            task = (
                _target_task(series)
                if suitable
                else "undetermined"
            )

        except ValueError:
            task = "undetermined"

        recommendations.append({
            "column": name,
            "score": round(score, 2),
            "task_suggestion": task,
            "unique_values": unique_count,
            "missing_percent": round(
                missing_ratio * 100,
                2,
            ),
            "suitable_candidate": suitable,
            "reasons": reasons,
        })

    recommendations.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    candidates = [
        item
        for item in recommendations
        if item["suitable_candidate"]
    ]

    return _clean_json({
        "can_train": (
            row_count >= 30
            and bool(candidates)
        ),
        "rows": row_count,
        "columns": len(df.columns),
        "recommendations": recommendations[
            :max(1, int(limit))
        ],
        "message": (
            "Choose the column you want to predict. "
            "Ranking is heuristic and does not establish the correct "
            "business target."
            if candidates
            else (
                "No suitable target was detected. "
                "Review the dataset first."
            )
        ),
    })


# ============================================================
# 8. DATASET PREPARATION
# ============================================================

def _prepare_dataframe(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare model features and target without modifying the original data.

    Removes missing-target rows, constant columns and excessively
    high-cardinality text columns. Extracts year/month/day from columns
    that Pandas has already recognized as datetime.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )

    if df.empty or len(df.columns) < 2:
        raise ValueError(
            "Dataset must have rows and at least two columns."
        )

    if not df.columns.is_unique:
        raise ValueError(
            "Dataset column names must be unique."
        )

    working = df.copy()

    working = working.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    target = working[target_column]

    valid_target = target.notna()

    working = working.loc[
        valid_target
    ].copy()

    target = working.pop(target_column)

    if len(working) < 30:
        raise ValueError(
            "At least 30 rows with a non-missing target are required "
            "for train/test evaluation."
        )

    dropped_columns: list[str] = []

    # Remove empty and constant feature columns.
    for column in working.columns:
        if (
            working[column].notna().sum() == 0
            or working[column].nunique(dropna=True) <= 1
        ):
            dropped_columns.append(
                str(column)
            )

    working = working.drop(
        columns=dropped_columns,
        errors="ignore",
    )

    # Extract features from datetime-typed columns.
    for column in list(working.columns):
        if pd.api.types.is_datetime64_any_dtype(
            working[column]
        ):
            dates = pd.to_datetime(
                working[column],
                errors="coerce",
            )

            working[f"{column}_year"] = dates.dt.year
            working[f"{column}_month"] = dates.dt.month
            working[f"{column}_day"] = dates.dt.day

            working = working.drop(
                columns=[column]
            )

    # Remove high-cardinality text features.
    for column in list(working.columns):
        if (
            not pd.api.types.is_numeric_dtype(
                working[column]
            )
            and working[column].nunique(dropna=True)
            > MAX_CATEGORIES
        ):
            dropped_columns.append(
                str(column)
            )

            working = working.drop(
                columns=[column]
            )

    if working.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after preprocessing."
        )

    # Convert numeric-looking text columns when at least 90% of
    # their non-missing values can be converted.
    for column in list(working.columns):
        if not pd.api.types.is_numeric_dtype(
            working[column]
        ):
            converted = pd.to_numeric(
                working[column],
                errors="coerce",
            )

            original_count = int(
                working[column].notna().sum()
            )

            converted_count = int(
                converted.notna().sum()
            )

            if (
                original_count > 0
                and converted_count / original_count >= 0.90
            ):
                working[column] = converted

    # Normalize categorical columns.
    for column in list(working.columns):
        if not pd.api.types.is_numeric_dtype(
            working[column]
        ):
            working[column] = (
                working[column]
                .astype("string")
                .astype(object)
            )

    return working, target, dropped_columns


# ============================================================
# 9. PREPROCESSING PIPELINE
# ============================================================

def _build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create preprocessing pipelines.

    Numeric:
        Median imputation and standardization.

    Categorical:
        Most-frequent imputation and one-hot encoding.

    The preprocessing is fitted separately inside each model pipeline
    to avoid fitting imputers/scalers/encoders on the held-out test data.
    """

    numeric_columns = list(
        X.select_dtypes(
            include=["number", "bool"]
        ).columns
    )

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=50,
                ),
            ),
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ))

    if not transformers:
        raise ValueError(
            "No numeric or categorical features are available."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# 10. CANDIDATE MACHINE-LEARNING MODELS
# ============================================================

def _model_candidates(
    task: str,
    row_count: int,
) -> list[tuple[str, Any]]:
    """Return candidate models for classification or regression."""

    if task == "classification":
        models: list[tuple[str, Any]] = [
            (
                "Logistic Regression",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=150,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "Extra Trees",
                ExtraTreesClassifier(
                    n_estimators=150,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]

        if row_count <= MAX_ROWS_FOR_NEURAL_NETWORK:
            models.append((
                "Neural Network (MLP)",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    early_stopping=True,
                    max_iter=250,
                    random_state=RANDOM_STATE,
                ),
            ))

        return models

    models = [
        (
            "Ridge Regression",
            Ridge(),
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "Extra Trees",
            ExtraTreesRegressor(
                n_estimators=150,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    if row_count <= MAX_ROWS_FOR_NEURAL_NETWORK:
        models.append((
            "Neural Network (MLP)",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                max_iter=250,
                random_state=RANDOM_STATE,
            ),
        ))

    return models


# ============================================================
# 11. POSSIBLE TARGET-LEAKAGE WARNINGS
# ============================================================

def _target_leakage_warnings(
    feature_columns: list[str],
    target_column: str,
) -> list[str]:
    """
    Flag some common feature names that might reveal the target.

    This is a heuristic check. It cannot detect every type of leakage.
    """

    warnings: list[str] = []

    def normalize(value: str) -> str:
        return " ".join(
            str(value)
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .split()
        )

    normalized_target = normalize(
        target_column
    )

    normalized_features = {
        normalize(column)
        for column in feature_columns
    }

    price_columns = {
        "price",
        "unit price",
        "price per unit",
    }

    quantity_columns = {
        "quantity",
        "qty",
        "units sold",
    }

    amount_targets = {
        "total spent",
        "total spend",
        "total sales",
        "total revenue",
        "sales amount",
        "total amount",
        "revenue",
    }

    if (
        normalized_target in amount_targets
        and normalized_features.intersection(
            price_columns
        )
        and normalized_features.intersection(
            quantity_columns
        )
    ):
        warnings.append(
            "Possible target leakage or mathematically derived target. "
            "Check whether the target is calculated from price and quantity."
        )

    target_words = set(
        normalized_target.split()
    )

    for column in normalized_features:
        feature_words = set(
            column.split()
        )

        if (
            column == normalized_target
            or (
                len(target_words) >= 2
                and target_words.issubset(feature_words)
            )
        ):
            warnings.append(
                f"Feature '{column}' has a name similar to target "
                f"'{target_column}'. Check whether it reveals information "
                "that would not be available during real predictions."
            )

    return warnings


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

def _feature_importance(
    pipeline: Pipeline,
    feature_columns: list[str],
) -> dict[str, Any]:
    """
    Extract model feature importance or absolute coefficients.

    For one-hot encoded data, the reported names may represent encoded
    categories rather than the original source columns.
    """

    try:
        preprocessor = pipeline.named_steps[
            "preprocessor"
        ]

        model = pipeline.named_steps[
            "model"
        ]

        transformed_names = (
            preprocessor.get_feature_names_out()
        )

        importance = None
        importance_type = None

        if hasattr(
            model,
            "feature_importances_",
        ):
            importance = np.asarray(
                model.feature_importances_
            )

            importance_type = (
                "tree_feature_importance"
            )

        elif hasattr(
            model,
            "coef_",
        ):
            coefficients = np.asarray(
                model.coef_
            )

            if coefficients.ndim == 1:
                importance = np.abs(
                    coefficients
                )
            else:
                importance = np.mean(
                    np.abs(coefficients),
                    axis=0,
                )

            importance_type = (
                "absolute_model_coefficients"
            )

        if importance is None:
            return {
                "available": False,
                "method": None,
                "features": [],
            }

        if len(transformed_names) != len(importance):
            return {
                "available": False,
                "method": None,
                "features": [],
            }

        entries = [
            {
                "feature": str(name),
                "importance": float(value),
            }
            for name, value in zip(
                transformed_names,
                importance,
            )
        ]

        entries.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        return _clean_json({
            "available": True,
            "method": importance_type,
            "features": entries[:100],
            "source_features": feature_columns,
        })

    except Exception:
        return {
            "available": False,
            "method": None,
            "features": [],
        }


# ============================================================
# 13. MAIN TRAINING AND EVALUATION FUNCTION
# ============================================================

def train_and_evaluate(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "auto",
    test_size: float = 0.2,
    return_model: bool = False,
) -> Any:
    """
    Compare models, select the strongest validation candidate,
    and evaluate the selected model on a held-out test split.

    Classification:
        Accuracy and weighted F1.

    Regression:
        MAE, RMSE, and R².

    Returns a JSON-safe result dictionary by default.

    With return_model=True, returns:
        (result, fitted_pipeline, feature_columns)
    """

    try:
        test_size = float(test_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "test_size must be a number between 0.10 and 0.40."
        ) from exc

    if not 0.10 <= test_size <= 0.40:
        raise ValueError(
            "test_size must be between 0.10 and 0.40."
        )

    X, y, dropped_columns = _prepare_dataframe(
        df,
        target_column,
    )

    requested_task = str(
        task_type
    ).strip().lower()

    if requested_task not in {
        "auto",
        "classification",
        "regression",
    }:
        raise ValueError(
            "task_type must be auto, classification, or regression."
        )

    inferred_task = _target_task(y)

    task = (
        inferred_task
        if requested_task == "auto"
        else requested_task
    )

    # --------------------------------------------------------
    # Classification target validation
    # --------------------------------------------------------

    if task == "classification":
        y = y.astype(str)

        class_counts = y.value_counts()

        if len(class_counts) < 2:
            raise ValueError(
                "Classification requires at least two classes."
            )

        if int(class_counts.min()) < 4:
            raise ValueError(
                "Each class needs at least four examples for evaluation."
            )

        if len(class_counts) > MAX_TARGET_CLASSES:
            raise ValueError(
                "This target has too many distinct classes for the "
                "current classification workflow."
            )

    # --------------------------------------------------------
    # Regression target validation
    # --------------------------------------------------------

    else:
        y = pd.to_numeric(
            y,
            errors="coerce",
        )

        valid = y.notna()

        X = X.loc[valid]
        y = y.loc[valid]

        if y.nunique() < 2:
            raise ValueError(
                "Regression requires at least two distinct numeric values."
            )

    if len(X) < 30:
        raise ValueError(
            "At least 30 usable rows are required."
        )

    # --------------------------------------------------------
    # Create a final held-out test split
    # --------------------------------------------------------

    stratify = (
        y
        if task == "classification"
        else None
    )

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )

    except ValueError as exc:
        raise ValueError(
            "Could not create a valid train/test split. "
            "Try a larger dataset or review class frequencies."
        ) from exc

    if (
        task == "classification"
        and y_train.nunique() < 2
    ):
        raise ValueError(
            "Training split must contain at least two classes."
        )

    # --------------------------------------------------------
    # Create a validation split from training data only
    # --------------------------------------------------------

    try:
        (
            X_fit,
            X_val,
            y_fit,
            y_val,
        ) = train_test_split(
            X_train,
            y_train,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=(
                y_train
                if task == "classification"
                else None
            ),
        )

    except ValueError:
        try:
            (
                X_fit,
                X_val,
                y_fit,
                y_val,
            ) = train_test_split(
                X_train,
                y_train,
                test_size=0.25,
                random_state=RANDOM_STATE,
                stratify=None,
            )

        except ValueError as exc:
            raise ValueError(
                "Could not create a separate validation split. "
                "Try a larger dataset."
            ) from exc

    # --------------------------------------------------------
    # Train and compare candidate models
    # --------------------------------------------------------

    candidates = _model_candidates(
        task,
        len(X),
    )

    comparison: list[dict[str, Any]] = []

    for name, estimator in candidates:
        try:
            pipeline = Pipeline([
                (
                    "preprocessor",
                    _build_preprocessor(X_fit),
                ),
                (
                    "model",
                    clone(estimator),
                ),
            ])

            pipeline.fit(
                X_fit,
                y_fit,
            )

            predicted = pipeline.predict(
                X_val
            )

            if task == "classification":
                score = float(
                    f1_score(
                        y_val,
                        predicted,
                        average="weighted",
                        zero_division=0,
                    )
                )

                metrics = {
                    "validation_weighted_f1": score,
                    "validation_accuracy": float(
                        accuracy_score(
                            y_val,
                            predicted,
                        )
                    ),
                }

                selection_score = score

            else:
                rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            y_val,
                            predicted,
                        )
                    )
                )

                mae = float(
                    mean_absolute_error(
                        y_val,
                        predicted,
                    )
                )

                metrics = {
                    "validation_rmse": rmse,
                    "validation_mae": mae,
                }

                # A smaller RMSE is better.
                selection_score = -rmse

            comparison.append({
                "model": name,
                "status": "success",
                "selection_score": selection_score,
                "metrics": metrics,
            })

        except Exception as exc:
            comparison.append({
                "model": name,
                "status": "failed",
                "reason": (
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:250]}"
                ),
            })

    successful = [
        item
        for item in comparison
        if item["status"] == "success"
    ]

    if not successful:
        raise ValueError(
            "All candidate models failed. Check target and feature columns."
        )

    best_row = max(
        successful,
        key=lambda item: item["selection_score"],
    )

    best_name = best_row["model"]

    estimator_by_name = dict(
        candidates
    )

    # --------------------------------------------------------
    # Refit the selected model using all training rows
    # --------------------------------------------------------

    final_model = Pipeline([
        (
            "preprocessor",
            _build_preprocessor(X_train),
        ),
        (
            "model",
            clone(
                estimator_by_name[best_name]
            ),
        ),
    ])

    final_model.fit(
        X_train,
        y_train,
    )

    # The test split is used only after model selection.
    predictions = final_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Calculate held-out test metrics
    # --------------------------------------------------------

    if task == "classification":
        final_metrics: dict[str, Any] = {
            "accuracy": float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

        labels = sorted(
            pd.unique(
                pd.concat([
                    y_train,
                    y_test,
                ])
            ).tolist()
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        )

        classification_details = {
            "labels": [
                str(label)
                for label in labels
            ],
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
        }

    else:
        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )
        )

        mae = float(
            mean_absolute_error(
                y_test,
                predictions,
            )
        )

        final_metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": (
                float(
                    r2_score(
                        y_test,
                        predictions,
                    )
                )
                if len(y_test) >= 2
                else None
            ),
        }

        classification_details = None

    # --------------------------------------------------------
    # Generate a small prediction preview
    # --------------------------------------------------------

    preview = []

    for actual, predicted in list(
        zip(
            y_test.tolist(),
            predictions.tolist(),
        )
    )[:100]:
        preview.append({
            "actual": _clean_json(actual),
            "predicted": _clean_json(predicted),
        })

    # --------------------------------------------------------
    # Warnings and feature importance
    # --------------------------------------------------------

    warnings = [
        (
            "Metrics are estimates from this dataset's held-out test split."
        ),
        (
            "Performance can change on new data and does not prove causation."
        ),
    ]

    warnings.extend(
        _target_leakage_warnings(
            [
                str(column)
                for column in X.columns
            ],
            target_column,
        )
    )

    if len(X) < 100:
        warnings.append(
            "The dataset is small; metrics may vary substantially "
            "across different samples."
        )

    if best_name == "Neural Network (MLP)":
        warnings.append(
            "A scikit-learn multilayer perceptron was selected. "
            "It is a basic neural-network model, not a GPU-based "
            "deep-learning pipeline."
        )

    importance = _feature_importance(
        final_model,
        [
            str(column)
            for column in X.columns
        ],
    )

    feature_columns = [
        str(column)
        for column in X.columns
    ]

    # --------------------------------------------------------
    # Build final result
    # --------------------------------------------------------

    result = _clean_json({
        "task": task,
        "target_column": target_column,

        "rows_used": int(len(X)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),

        "features_used": feature_columns,

        "dropped_constant_or_empty_features": (
            dropped_columns
        ),

        "selected_model": best_name,

        "selection_metric": (
            "validation_weighted_f1"
            if task == "classification"
            else "lowest_validation_rmse"
        ),

        "validation_metrics_for_selected_model": (
            best_row["metrics"]
        ),

        "test_metrics": final_metrics,

        "model_comparison": comparison,

        "prediction_preview": preview,

        "classification_details": (
            classification_details
        ),

        "feature_importance": importance,

        "warnings": warnings,

        "deep_learning": {
            "framework": "scikit-learn MLP",
            "used": (
                best_name == "Neural Network (MLP)"
            ),
            "note": (
                "The neural-network candidate was evaluated when the "
                "dataset size was within the configured limit. "
                "PyTorch is not required."
            ),
        },
    })

    # Keep the original return contract for the existing API.
    if return_model:
        return (
            result,
            final_model,
            feature_columns,
        )

    return result
