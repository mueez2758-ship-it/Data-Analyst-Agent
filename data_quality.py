
# ============================================================
# PROJECT: Data Analyst Agent
# FILE: app/data_quality.py
# PURPOSE: Detect common data quality problems.
# ============================================================

import numpy as np
import pandas as pd


def assess_data_quality(df: pd.DataFrame) -> dict:
    """
    Inspect a dataset and return a data quality report.

    df: The dataset to inspect.
    Returns: A dictionary containing detected issues.
    """

    # Count the rows and columns in the dataset.
    total_rows = len(df)
    total_columns = len(df.columns)

    # Create lists to store warnings and recommendations.
    warnings = []
    recommendations = []

    # --------------------------------------------------------
    # STEP 1: CHECK MISSING VALUES
    # --------------------------------------------------------

    # Count missing values in every column.
    missing_counts = df.isna().sum()

    missing_report = {}

    for column in df.columns:
        missing_count = int(missing_counts[column])

        # Calculate the percentage of missing values.
        missing_percentage = (
            (missing_count / total_rows) * 100
            if total_rows > 0
            else 0
        )

        # Save the missing-value information.
        missing_report[str(column)] = {
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage, 2
            ),
        }

        # Add a warning if missing values exist.
        if missing_count > 0:
            warnings.append(
                f"Column '{column}' contains "
                f"{missing_count} missing values."
            )

            recommendations.append(
                f"Review missing values in '{column}' "
                "before analysis."
            )

    # --------------------------------------------------------
    # STEP 2: CHECK DUPLICATE ROWS
    # --------------------------------------------------------

    # Count rows that duplicate earlier complete rows.
    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows > 0:
        warnings.append(
            f"Found {duplicate_rows} duplicate rows."
        )

        recommendations.append(
            "Review duplicate rows before removing them."
        )

    # --------------------------------------------------------
    # STEP 3: CHECK CONSTANT COLUMNS
    # --------------------------------------------------------

    # A constant column contains at most one distinct value.
    constant_columns = []

    for column in df.columns:
        unique_count = df[column].nunique(dropna=False)

        if unique_count <= 1:
            constant_columns.append(str(column))

            warnings.append(
                f"Column '{column}' has at most one "
                "distinct value."
            )

            recommendations.append(
                f"Check whether '{column}' is useful "
                "for your analysis."
            )

    # --------------------------------------------------------
    # STEP 4: DETECT POTENTIAL NUMERICAL OUTLIERS
    # --------------------------------------------------------

    # IQR is the interquartile range.
    # Potential outliers lie outside:
    # Q1 - 1.5 * IQR or Q3 + 1.5 * IQR.
    #
    # Outliers are not automatically errors.

    numeric_df = df.select_dtypes(include="number")
    outlier_report = {}

    for column in numeric_df.columns:

        # Ignore missing and infinite values.
        values = numeric_df[column].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        # Too few values means the check may not be useful.
        if len(values) < 4:
            outlier_report[str(column)] = {
                "checked_values": int(len(values)),
                "outlier_count": 0,
                "status": "Insufficient valid values",
            }
            continue

        # Calculate the first and third quartiles.
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        # Calculate the interquartile range.
        iqr = q3 - q1

        # A zero IQR makes this rule uninformative.
        if iqr == 0:
            outlier_report[str(column)] = {
                "checked_values": int(len(values)),
                "outlier_count": 0,
                "status": "IQR is zero",
            }
            continue

        # Calculate the lower and upper boundaries.
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        # Identify values outside the boundaries.
        outliers = (
            (values < lower_bound)
            | (values > upper_bound)
        )

        outlier_count = int(outliers.sum())

        # Store the results for this column.
        outlier_report[str(column)] = {
            "checked_values": int(len(values)),
            "outlier_count": outlier_count,
            "lower_boundary": float(lower_bound),
            "upper_boundary": float(upper_bound),
            "method": "IQR",
        }

        if outlier_count > 0:
            warnings.append(
                f"Column '{column}' has "
                f"{outlier_count} potential outliers."
            )

            recommendations.append(
                f"Investigate extreme values in '{column}' "
                "before changing any data."
            )

    # --------------------------------------------------------
    # STEP 5: CHECK FOR EMPTY DATASETS
    # --------------------------------------------------------

    if total_rows == 0:
        warnings.append("The dataset has no data rows.")

        recommendations.append(
            "Upload a dataset containing valid records."
        )

    if total_columns == 0:
        warnings.append("The dataset has no columns.")

    # --------------------------------------------------------
    # STEP 6: DETERMINE THE QUALITY STATUS
    # --------------------------------------------------------

    if total_rows == 0 or total_columns == 0:
        quality_status = "invalid_or_empty"

    elif warnings:
        quality_status = "review_recommended"

    else:
        quality_status = "no_common_issues_detected"

    # --------------------------------------------------------
    # STEP 7: RETURN THE FINAL QUALITY REPORT
    # --------------------------------------------------------

    return {
        "rows_checked": int(total_rows),
        "columns_checked": int(total_columns),
        "quality_status": quality_status,
        "missing_values": missing_report,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "potential_outliers": outlier_report,
        "warnings": warnings,
        "recommendations": recommendations,
    }
"""Check common data-quality issues without changing the dataset."""

import numpy as np
import pandas as pd


def assess_data_quality(df: pd.DataFrame) -> dict:
    """Check missing values, duplicates, constants, and outliers."""

    rows = int(len(df))
    columns = int(len(df.columns))

    # Count missing values in every column.
    missing_values = {}

    for column in df.columns:
        count = int(df[column].isna().sum())

        missing_values[str(column)] = {
            "missing_count": count,
            "missing_percentage": (
                round(count / rows * 100, 2) if rows else 0
            ),
        }

    # Count duplicate rows.
    duplicate_rows = int(df.duplicated().sum())

    # Find columns containing only one distinct value.
    constant_columns = [
        str(column)
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    # Detect potential numerical outliers using the IQR method.
    potential_outliers = {}

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )
        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if values.empty:
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = int(
            ((values < lower) | (values > upper)).sum()
        )

        potential_outliers[str(column)] = {
            "checked_values": int(values.count()),
            "outlier_count": outlier_count,
            "lower_boundary": lower,
            "upper_boundary": upper,
            "method": "IQR",
        }

    # Prepare warnings and recommendations.
    warnings = []
    recommendations = []

    total_missing = sum(
        item["missing_count"]
        for item in missing_values.values()
    )

    if rows == 0:
        quality_status = "invalid_or_empty"
        warnings.append("The dataset contains no data rows.")
        recommendations.append(
            "Upload a dataset containing at least one data row."
        )
    elif total_missing or duplicate_rows or constant_columns:
        quality_status = "review_recommended"
    else:
        quality_status = "no_common_issues_detected"

    if total_missing:
        warnings.append(
            f"{total_missing} missing cell(s) were detected."
        )
        recommendations.append(
            "Review missing values before downstream analysis."
        )

    if duplicate_rows:
        warnings.append(
            f"{duplicate_rows} duplicate row(s) were detected."
        )
        recommendations.append(
            "Review duplicates before removing any records."
        )

    if constant_columns:
        warnings.append(
            "One or more columns contain only one distinct value."
        )

    return {
        "rows_checked": rows,
        "columns_checked": columns,
        "quality_status": quality_status,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "potential_outliers": potential_outliers,
        "warnings": warnings,
        "recommendations": recommendations,
    }