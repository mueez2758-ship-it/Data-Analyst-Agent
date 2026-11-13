
# ============================================================
# PROJECT: Data Analyst Agent
# FILE: app/main.py
# VERSION: 0.3.3
#
# FEATURES:
# 1. FastAPI application
# 2. CSV, XLSX, and XLS dataset uploads
# 3. Statistical analysis
# 4. Data quality checks
# 5. Automated insights and recommendations
# 6. Automatic dashboard KPIs and charts
# 7. Structured JSON responses
# 8. Upload validation and safe HTTP errors
# ============================================================

from io import BytesIO
from pathlib import Path
from typing import Any
import logging
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.dashboard import generate_dashboard
from app.data_quality import assess_data_quality
from app.insights import generate_insights


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Analyst Agent",
    description=(
        "Automated data analysis, data quality checks, "
        "insights, predictions, SQL, and dashboards."
    ),
    version="0.3.3",
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# ============================================================
# 2. RESPONSE MODEL
# ============================================================

class DatasetAnalysisResponse(BaseModel):
    """Structure of a successful dataset analysis response."""

    filename: str
    file_size_bytes: int
    analysis: dict[str, Any]
    data_quality: dict[str, Any]
    insights: dict[str, Any]
    dashboard: dict[str, Any]
    notice: str


# ============================================================
# 3. CONVERT VALUES TO JSON-SAFE VALUES
# ============================================================

def clean_value(value: Any) -> Any:
    """Convert NumPy and Pandas values into JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


# ============================================================
# 4. STATISTICAL ANALYSIS
# ============================================================

def analyze_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Generate statistical summaries for a dataset."""

    numeric_df = df.select_dtypes(include="number")

    categorical_df = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    )

    numeric_summary: dict[str, Any] = {}
    categorical_summary: dict[str, Any] = {}

    # Numerical statistics
    for column in numeric_df.columns:
        series = pd.to_numeric(
            numeric_df[column],
            errors="coerce",
        )

        series = series.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        numeric_summary[str(column)] = {
            "count": int(series.count()),
            "mean": clean_value(series.mean()),
            "median": clean_value(series.median()),
            "minimum": clean_value(series.min()),
            "maximum": clean_value(series.max()),
            "standard_deviation": clean_value(series.std()),
        }

    # Categorical summaries
    for column in categorical_df.columns:
        top_values = (
            df[column]
            .dropna()
            .astype(str)
            .value_counts()
            .head(5)
        )

        categorical_summary[str(column)] = [
            {
                "value": str(value),
                "count": int(count),
            }
            for value, count in top_values.items()
        ]

    # Missing values
    missing_values = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }

    # Column data types
    column_types = {
        str(column): str(dtype)
        for column, dtype in df.dtypes.items()
    }

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # Initial findings
    findings: list[str] = []

    if df.empty:
        findings.append("The dataset contains no data rows.")

    if missing_values:
        findings.append(
            f"{len(missing_values)} column(s) contain missing values."
        )
    else:
        findings.append("No missing values were detected.")

    if duplicate_count:
        findings.append(
            f"{duplicate_count} duplicate row(s) were detected."
        )
    else:
        findings.append("No duplicate rows were detected.")

    if len(numeric_df.columns) > 0:
        findings.append(
            f"{len(numeric_df.columns)} numerical column(s) "
            "are available for statistical analysis."
        )

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(column) for column in df.columns],
        "column_types": column_types,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_count,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "initial_findings": findings,
    }


# ============================================================
# 5. HOME ENDPOINT
# ============================================================

@app.get("/", summary="Home")
def home() -> dict[str, str]:
    """Return API information."""

    return {
        "message": "Welcome to Data Analyst Agent",
        "status": "running",
        "version": app.version,
        "docs": "/docs",
    }


# ============================================================
# 6. DATASET ANALYSIS ENDPOINT
# ============================================================

@app.post(
    "/api/v1/datasets/analyze",
    response_model=DatasetAnalysisResponse,
    summary="Analyze Dataset",
    description=(
        "Upload a CSV or Excel file to generate statistical "
        "analysis, data quality checks, automated insights, "
        "KPIs, and dashboard charts."
    ),
    responses={
        400: {"description": "Invalid or unsupported dataset."},
        413: {"description": "Uploaded file exceeds 25 MB."},
        500: {"description": "Dataset analysis failed."},
    },
)
async def analyze_dataset(
    file: UploadFile = File(...),
) -> DatasetAnalysisResponse:
    """Analyze an uploaded dataset and return all reports."""

    try:
        # Validate filename and extension
        filename = Path(file.filename or "").name
        extension = Path(filename).suffix.lower()

        if not filename or extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Upload a CSV, XLSX, or XLS file.",
            )

        # Read upload in chunks with a size limit
        chunks: list[bytes] = []
        total_bytes = 0

        while True:
            chunk = await file.read(1024 * 1024)

            if not chunk:
                break

            total_bytes += len(chunk)

            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="File exceeds the 25 MB upload limit.",
                )

            chunks.append(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        content = b"".join(chunks)

        # Load CSV or Excel
        try:
            if extension == ".csv":
                df = pd.read_csv(BytesIO(content))

            elif extension == ".xlsx":
                df = pd.read_excel(
                    BytesIO(content),
                    engine="openpyxl",
                )

            else:
                df = pd.read_excel(
                    BytesIO(content),
                    engine="xlrd",
                )

        except ImportError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "An Excel dependency is missing. Install "
                    "openpyxl for XLSX or xlrd for XLS files."
                ),
            ) from exc

        except (
            UnicodeDecodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            ValueError,
            OSError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not read the file. Check that it is "
                    "a valid CSV or Excel dataset."
                ),
            ) from exc

        # Validate dataset structure
        if len(df.columns) == 0:
            raise HTTPException(
                status_code=400,
                detail="The dataset contains no columns.",
            )

        if not df.columns.is_unique:
            raise HTTPException(
                status_code=400,
                detail="Column names must be unique.",
            )

        # Generate statistical analysis
        analysis_report = analyze_dataframe(df)

        # Assess data quality
        quality_report = assess_data_quality(df)

        # Generate insights and recommendations
        insights_report = generate_insights(df)

        # Generate dashboard KPIs and charts
        dashboard_report = generate_dashboard(df)

        # Return complete JSON response
        return DatasetAnalysisResponse(
            filename=filename,
            file_size_bytes=total_bytes,
            analysis=analysis_report,
            data_quality=quality_report,
            insights=insights_report,
            dashboard=dashboard_report,
            notice=(
                "These reports describe the uploaded data. "
                "They do not guarantee data correctness "
                "or prediction accuracy."
            ),
        )

    except HTTPException:
        # Preserve intended HTTP status codes.
        raise

    except Exception as exc:
        # Log unexpected errors for debugging.
        logger.exception("Dataset analysis failed")

        raise HTTPException(
            status_code=500,
            detail=(
                "Dataset analysis failed. "
                "Check the server terminal for details."
            ),
        ) from exc

    finally:
        # Always close the uploaded file.
        await file.close()

# Additional project and feature-engineering endpoints
from app.project_api import router as project_router
from app.feature_api import router as feature_router

app.include_router(project_router)
app.include_router(feature_router)
