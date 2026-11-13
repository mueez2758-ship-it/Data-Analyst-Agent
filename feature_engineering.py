
"""Feature engineering utilities for the Data Analyst Agent."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _unique_column_name(
    desired_name: str,
    existing_columns: set[str],
) -> str:
    """Return a column name that does not overwrite existing data."""
    name = desired_name
    counter = 1

    while name in existing_columns:
        name = f"{desired_name}_{counter}"
        counter += 1

    existing_columns.add(name)
    return name


def engineer_features(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = False,
    fill_numeric_missing: bool = False,
    fill_categorical_missing: bool = False,
    encode_categoricals: bool = False,
    extract_date_features: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transform a copy of a DataFrame and return a JSON-safe summary.

    The input DataFrame is never modified.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if not df.columns.is_unique:
        raise ValueError("Column names must be unique.")

    result = df.copy(deep=True)
    original_rows, original_columns = result.shape
    changes: list[str] = []

    # 1. Remove duplicate rows if requested.
    if drop_duplicates:
        before = len(result)
        result = result.drop_duplicates().reset_index(drop=True)
        removed = before - len(result)
        changes.append(f"Removed {removed} duplicate row(s).")

    # 2. Fill missing numeric values using each column's median.
    if fill_numeric_missing:
        numeric_columns = result.select_dtypes(
            include="number"
        ).columns

        filled_columns: list[str] = []

        for column in numeric_columns:
            series = result[column].replace(
                [float("inf"), float("-inf")],
                float("nan"),
            )

            if series.isna().any():
                median = series.median()

                if pd.notna(median):
                    result[column] = series.fillna(median)
                    filled_columns.append(str(column))

        changes.append(
            "Filled numeric missing values with medians in: "
            + (
                ", ".join(filled_columns)
                if filled_columns
                else "no eligible columns"
            )
            + "."
        )

    # 3. Fill missing categorical values.
    if fill_categorical_missing:
        categorical_columns = result.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns

        filled_columns = []

        for column in categorical_columns:
            if result[column].isna().any():
                # Convert to object so categorical columns can accept
                # the new "Unknown" category safely.
                result[column] = (
                    result[column]
                    .astype("object")
                    .fillna("Unknown")
                )
                filled_columns.append(str(column))

        changes.append(
            "Filled categorical missing values in: "
            + (
                ", ".join(filled_columns)
                if filled_columns
                else "no columns needed filling"
            )
            + "."
        )

    # 4. Extract date components from columns that appear to be dates.
    if extract_date_features:
        candidates = list(
            result.select_dtypes(
                include=["object", "string"]
            ).columns
        )

        created_columns: list[str] = []
        existing_columns = {str(column) for column in result.columns}

        for column in candidates:
            values = result[column]
            parsed = pd.to_datetime(values, errors="coerce")

            if len(parsed) == 0:
                continue

            # Avoid treating ordinary text as a date column.
            if parsed.notna().mean() < 0.8:
                continue

            prefix = str(column)[:40]

            date_parts = {
                "year": parsed.dt.year,
                "month": parsed.dt.month,
                "day": parsed.dt.day,
                "day_of_week": parsed.dt.dayofweek,
            }

            for part_name, values_series in date_parts.items():
                desired_name = f"{prefix}_{part_name}"
                new_name = _unique_column_name(
                    desired_name,
                    existing_columns,
                )
                result[new_name] = values_series
                created_columns.append(new_name)

        changes.append(
            "Created date features: "
            + (
                ", ".join(created_columns)
                if created_columns
                else "no columns met the date-detection threshold"
            )
            + "."
        )

    # 5. One-hot encode categorical columns.
    if encode_categoricals:
        categorical_columns = list(
            result.select_dtypes(
                include=["object", "category", "string", "bool"]
            ).columns
        )

        if categorical_columns:
            result = pd.get_dummies(
                result,
                columns=categorical_columns,
                dummy_na=False,
                dtype=int,
            )
            changes.append(
                f"One-hot encoded {len(categorical_columns)} "
                "categorical column(s)."
            )
        else:
            changes.append("No categorical columns needed encoding.")

    # Build a JSON-safe summary of the transformed data.
    missing_values = {
        str(column): int(count)
        for column, count in result.isna().sum().items()
        if count > 0
    }

    numeric_columns = [
        str(column)
        for column in result.select_dtypes(include="number").columns
    ]

    summary = {
        "original_shape": {
            "rows": int(original_rows),
            "columns": int(original_columns),
        },
        "transformed_shape": {
            "rows": int(result.shape[0]),
            "columns": int(result.shape[1]),
        },
        "columns": [str(column) for column in result.columns],
        "remaining_missing_values": missing_values,
        "numeric_columns": numeric_columns,
        "changes": changes,
        "note": (
            "Transformations were applied to a copy. Review the "
            "resulting dataset before using it for machine learning. "
            "For predictive modeling, fit preprocessing steps using "
            "training data only to avoid data leakage."
        ),
    }

    return result, summary
