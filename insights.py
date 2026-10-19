
# ============================================================
# PROJECT: Data Analyst Agent
# FILE: app/insights.py
# VERSION: 0.4.0
# PURPOSE:
# 1. Generate automatic descriptive insights.
# 2. Identify missing values and duplicate rows.
# 3. Analyze numerical distributions.
# 4. Discover useful categorical patterns.
# 5. Compare numerical metrics across categories.
# 6. Detect strong numerical correlations.
# 7. Generate data-driven recommendations.
# ============================================================

from typing import Any

import math
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

MAX_FINDINGS = 30
MAX_RECOMMENDATIONS = 20
MAX_CATEGORIES_TO_ANALYZE = 20
MIN_ROWS_FOR_COMPARISON = 3


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def _is_finite_number(value: Any) -> bool:
    """Return True if a value is a finite number."""

    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _format_number(value: Any) -> str:
    """Format a finite numeric value for readable reports."""

    if not _is_finite_number(value):
        return "unavailable"

    number = float(value)

    if number.is_integer():
        return f"{number:,.0f}"

    return f"{number:,.2f}"


def _safe_text(value: Any, limit: int = 100) -> str:
    """Convert a value to a short, safe display string."""

    text = str(value).replace("\n", " ").strip()

    if len(text) > limit:
        return text[: limit - 3] + "..."

    return text


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns, excluding Boolean columns."""

    return [
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return suitable text and category columns."""

    return [
        str(column)
        for column in df.columns
        if (
            pd.api.types.is_object_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
        )
    ]


def _add_recommendation(
    recommendations: list[dict[str, str]],
    title: str,
    message: str,
    priority: str = "medium",
) -> None:
    """Add a recommendation without creating duplicates."""

    if len(recommendations) >= MAX_RECOMMENDATIONS:
        return

    if any(item["title"] == title for item in recommendations):
        return

    recommendations.append(
        {
            "title": title,
            "priority": priority,
            "recommendation": message,
        }
    )


# ============================================================
# 3. DATASET OVERVIEW
# ============================================================

def _analyze_overview(
    df: pd.DataFrame,
    findings: list[str],
    recommendations: list[dict[str, str]],
) -> None:
    """Describe dataset dimensions and general structure."""

    rows, columns = df.shape

    findings.append(
        f"The dataset contains {rows:,} rows and {columns:,} columns."
    )

    if rows == 0:
        findings.append(
            "The dataset has no data rows, so statistical conclusions "
            "cannot be generated."
        )

        _add_recommendation(
            recommendations,
            "Upload a non-empty dataset",
            "Provide a dataset containing valid observations before "
            "running statistical analysis.",
            "high",
        )

    if columns == 0:
        _add_recommendation(
            recommendations,
            "Provide dataset columns",
            "Ensure the uploaded file contains at least one named column.",
            "high",
        )

    if rows > 0 and columns > 0:
        memory_bytes = int(df.memory_usage(deep=True).sum())

        findings.append(
            f"The loaded dataset uses approximately "
            f"{memory_bytes / (1024 * 1024):.2f} MB of memory."
        )


# ============================================================
# 4. MISSING VALUES
# ============================================================

def _analyze_missing_values(
    df: pd.DataFrame,
    findings: list[str],
    recommendations: list[dict[str, str]],
) -> None:
    """Find columns containing missing data."""

    rows = len(df)

    if rows == 0:
        return

    missing_counts = df.isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0]

    if columns_with_missing.empty:
        findings.append("No missing values were detected.")
        return

    total_missing = int(columns_with_missing.sum())

    findings.append(
        f"{total_missing:,} missing cell(s) were detected across "
        f"{len(columns_with_missing)} column(s)."
    )

    for column, count in columns_with_missing.items():
        percentage = float(count / rows * 100)

        findings.append(
            f"'{_safe_text(column)}' contains {int(count):,} missing "
            f"value(s), representing {percentage:.1f}% of its rows."
        )

        if percentage >= 20:
            priority = "high"
        elif percentage >= 5:
            priority = "medium"
        else:
            priority = "low"

        _add_recommendation(
            recommendations,
            f"Review missing values in {_safe_text(column)}",
            f"Inspect the {int(count):,} missing value(s) in "
            f"'{_safe_text(column)}'. Determine why they are missing "
            "before choosing whether to correct, impute, or retain them.",
            priority,
        )


# ============================================================
# 5. DUPLICATE ROWS
# ============================================================

def _analyze_duplicates(
    df: pd.DataFrame,
    findings: list[str],
    recommendations: list[dict[str, str]],
) -> None:
    """Detect duplicate rows."""

    if df.empty:
        return

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        findings.append("No duplicate rows were detected.")
        return

    percentage = duplicate_count / len(df) * 100

    findings.append(
        f"{duplicate_count:,} duplicate row(s) were detected "
        f"({percentage:.1f}% of all rows)."
    )

    _add_recommendation(
        recommendations,
        "Review duplicate rows",
        f"Inspect the {duplicate_count:,} duplicate row(s). Confirm "
        "whether they represent repeated records or legitimate "
        "transactions before removing them.",
        "high" if percentage >= 10 else "medium",
    )


# ============================================================
# 6. NUMERICAL DISTRIBUTIONS
# ============================================================

def _analyze_numeric_columns(
    df: pd.DataFrame,
    findings: list[str],
    recommendations: list[dict[str, str]],
) -> None:
    """Analyze numeric ranges, variability and potential issues."""

    columns = _numeric_columns(df)

    if not columns:
        findings.append(
            "No numeric columns were identified for numerical analysis."
        )

        return

    findings.append(
        f"{len(columns)} numeric column(s) are available for analysis."
    )

    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")

        # Identify invalid infinite values separately.
        infinite_count = int(np.isinf(series.to_numpy(dtype=float)).sum())

        # Exclude infinities from descriptive statistics.
        valid = series.replace([np.inf, -np.inf], np.nan).dropna()

        if valid.empty:
            findings.append(
                f"'{_safe_text(column)}' has no finite numeric values."
            )

            _add_recommendation(
                recommendations,
                f"Review numeric column {_safe_text(column)}",
                f"Check '{_safe_text(column)}' for invalid or missing "
                "numeric values before using it in calculations.",
                "high",
            )

            continue

        minimum = float(valid.min())
        maximum = float(valid.max())
        mean = float(valid.mean())
        median = float(valid.median())

        findings.append(
            f"'{_safe_text(column)}' ranges from "
            f"{_format_number(minimum)} to {_format_number(maximum)}; "
            f"its mean is {_format_number(mean)} and median is "
            f"{_format_number(median)}."
        )

        if infinite_count:
            findings.append(
                f"'{_safe_text(column)}' contains {infinite_count} "
                "infinite value(s), which were excluded from summaries."
            )

            _add_recommendation(
                recommendations,
                f"Review infinite values in {_safe_text(column)}",
                f"Investigate the {infinite_count} infinite value(s) "
                f"in '{_safe_text(column)}' and determine appropriate "
                "corrections before downstream analysis.",
                "high",
            )

        if valid.nunique() <= 1 and len(valid) > 0:
            findings.append(
                f"'{_safe_text(column)}' has no observed numeric "
                "variation among its valid values."
            )

            _add_recommendation(
                recommendations,
                f"Review constant column {_safe_text(column)}",
                f"'{_safe_text(column)}' has only one distinct valid "
                "numeric value. Check whether the column is useful "
                "for the intended analysis.",
                "low",
            )


# ============================================================
# 7. CATEGORICAL PATTERNS
# ============================================================

def _analyze_categorical_columns(
    df: pd.DataFrame,
    findings: list[str],
    recommendations: list[dict[str, str]],
) -> None:
    """Identify frequent categories and possible identifier columns."""

    columns = _categorical_columns(df)

    for column in columns:
        series = df[column].dropna().astype(str)

        if series.empty:
            continue

        counts = series.value_counts()
        unique_count = int(counts.size)
        valid_count = int(series.size)

        most_common = _safe_text(counts.index[0])
        most_common_count = int(counts.iloc[0])
        most_common_pct = most_common_count / valid_count * 100

        findings.append(
            f"In '{_safe_text(column)}', the most common value is "
            f"'{most_common}' ({most_common_count:,} records, "
            f"{most_common_pct:.1f}% of non-missing values)."
        )

        # Very high cardinality can indicate an ID-like field.
        if (
            valid_count >= 10
            and unique_count / valid_count >= 0.95
        ):
            findings.append(
                f"'{_safe_text(column)}' has many unique values "
                "relative to its non-missing records."
            )

            _add_recommendation(
                recommendations,
                f"Review high-cardinality column {_safe_text(column)}",
                f"Check whether '{_safe_text(column)}' is an identifier "
                "or free-text field. Its suitability for grouping or "
                "model features depends on its meaning.",
                "low",
            )


# ============================================================
# 8. CATEGORY-BASED NUMERIC COMPARISONS
# ============================================================

def _analyze_group_comparisons(
    df: pd.DataFrame,
    findings: list[str],
    recommendations: list[dict[str, str]],
) -> None:
    """Compare numeric means across suitable categorical groups."""

    numeric_columns = _numeric_columns(df)
    category_columns = _categorical_columns(df)

    if len(df) < MIN_ROWS_FOR_COMPARISON:
        return

    for category_column in category_columns:
        category_values = df[category_column]

        # Avoid grouping on extremely high-cardinality fields.
        unique_count = int(category_values.nunique(dropna=True))

        if unique_count < 2 or unique_count > MAX_CATEGORIES_TO_ANALYZE:
            continue

        for numeric_column in numeric_columns:
            values = pd.to_numeric(
                df[numeric_column],
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)

            working = pd.DataFrame(
                {
                    "category": category_values,
                    "value": values,
                }
            ).dropna()

            if working.empty:
                continue

            grouped = working.groupby(
                "category",
                observed=True,
            )["value"].agg(["mean", "count"])

            # Require at least two groups with usable values.
            grouped = grouped[grouped["count"] > 0]

            if len(grouped) < 2:
                continue

            highest_category = grouped["mean"].idxmax()
            lowest_category = grouped["mean"].idxmin()

            highest_mean = float(grouped.loc[highest_category, "mean"])
            lowest_mean = float(grouped.loc[lowest_category, "mean"])

            findings.append(
                f"Average '{_safe_text(numeric_column)}' is highest "
                f"in '{_safe_text(category_column)}' group "
                f"'{_safe_text(highest_category)}' "
                f"({_format_number(highest_mean)}) and lowest in "
                f"'{_safe_text(lowest_category)}' "
                f"({_format_number(lowest_mean)})."
            )

            if lowest_mean != 0:
                difference_pct = (
                    (highest_mean - lowest_mean)
                    / abs(lowest_mean)
                    * 100
                )

                if (
                    _is_finite_number(difference_pct)
                    and abs(difference_pct) >= 50
                ):
                    _add_recommendation(
                        recommendations,
                        f"Investigate {_safe_text(numeric_column)} "
                        f"across {_safe_text(category_column)} groups",
                        "Review differences between category averages "
                        "and check group sizes, data quality, and "
                        "business context before deciding on action.",
                        "medium",
                    )


# ============================================================
# 9. CORRELATION ANALYSIS
# ============================================================

def _analyze_correlations(
    df: pd.DataFrame,
    findings: list[str],
    recommendations: list[dict[str, str]],
) -> None:
    """Report strong pairwise numeric correlations."""

    numeric_columns = _numeric_columns(df)

    if len(numeric_columns) < 2 or len(df) < 3:
        return

    numeric_data = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove entirely empty or constant columns.
    numeric_data = numeric_data.dropna(axis=1, how="all")
    numeric_data = numeric_data.loc[
        :,
        numeric_data.nunique(dropna=True) > 1,
    ]

    if numeric_data.shape[1] < 2:
        return

    correlation_matrix = numeric_data.corr(
        method="pearson",
        min_periods=3,
    )

    strong_pairs = []

    for i, column_a in enumerate(correlation_matrix.columns):
        for column_b in correlation_matrix.columns[i + 1:]:
            correlation = correlation_matrix.loc[column_a, column_b]

            if pd.isna(correlation):
                continue

            value = float(correlation)

            if abs(value) >= 0.7:
                strong_pairs.append(
                    (str(column_a), str(column_b), value)
                )

    strong_pairs.sort(
        key=lambda item: abs(item[2]),
        reverse=True,
    )

    for column_a, column_b, value in strong_pairs[:5]:
        direction = "positive" if value > 0 else "negative"

        findings.append(
            f"'{_safe_text(column_a)}' and "
            f"'{_safe_text(column_b)}' have a strong "
            f"{direction} linear correlation "
            f"(Pearson r = {value:.2f})."
        )

        _add_recommendation(
            recommendations,
            f"Investigate relationship between "
            f"{_safe_text(column_a)} and {_safe_text(column_b)}",
            "Check whether this relationship is meaningful in the "
            "domain context. Correlation alone does not establish "
            "causation and may be influenced by other variables.",
            "low",
        )


# ============================================================
# 10. GENERATE THE COMPLETE INSIGHTS REPORT
# ============================================================

def generate_insights(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate descriptive insights and actionable recommendations.

    Output remains compatible with the existing API:
        {
            "summary": str,
            "findings": list[str],
            "recommendations": list[dict[str, str]]
        }
    """

    findings: list[str] = []
    recommendations: list[dict[str, str]] = []

    if not isinstance(df, pd.DataFrame):
        raise TypeError("generate_insights expects a Pandas DataFrame.")

    # Empty datasets need special handling.
    if df.empty:
        _analyze_overview(df, findings, recommendations)

        return {
            "summary": (
                "The dataset contains no data rows. "
                "No statistical insights could be generated."
            ),
            "findings": findings[:MAX_FINDINGS],
            "recommendations": recommendations[:MAX_RECOMMENDATIONS],
        }

    # Run descriptive analysis modules.
    _analyze_overview(df, findings, recommendations)

    _analyze_missing_values(
        df,
        findings,
        recommendations,
    )

    _analyze_duplicates(
        df,
        findings,
        recommendations,
    )

    _analyze_numeric_columns(
        df,
        findings,
        recommendations,
    )

    _analyze_categorical_columns(
        df,
        findings,
        recommendations,
    )

    _analyze_group_comparisons(
        df,
        findings,
        recommendations,
    )

    _analyze_correlations(
        df,
        findings,
        recommendations,
    )

    # Provide a useful recommendation when no issues are detected.
    if not recommendations:
        _add_recommendation(
            recommendations,
            "Continue validating data quality",
            "No immediate issues requiring recommendations were "
            "identified by these checks. Confirm that the dataset "
            "covers the intended population and time period, and "
            "validate important fields against their source.",
            "low",
        )

    # Keep the response compact.
    findings = findings[:MAX_FINDINGS]
    recommendations = recommendations[:MAX_RECOMMENDATIONS]

    summary = (
        f"Reviewed {len(df):,} rows and {len(df.columns):,} columns. "
        f"Generated {len(findings)} findings and "
        f"{len(recommendations)} recommendations."
    )

    return {
        "summary": summary,
        "findings": findings,
        "recommendations": recommendations,
    }
