
# ============================================================
# PROJECT: Data Analyst Agent
# FILE: app/main.py
# VERSION: 0.3.5
#
# FEATURES:
# 1. FastAPI application
# 2. CSV, XLSX, and XLS uploads
# 3. Statistical analysis
# 4. Data quality assessment
# 5. Automated insights and recommendations
# 6. Automatic dashboard generation
# 7. Persistent dataset-named projects and saved reports
# 8. New Project support
# 9. Project, feature-engineering, and prediction APIs
# 10. Upload validation and safe HTTP error handling
# ============================================================

from io import BytesIO
from pathlib import Path
from typing import Any
import logging
import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from app.dashboard import generate_dashboard
from app.data_quality import assess_data_quality
from app.insights import generate_insights
from app.project_store import (
    create_project,
    get_or_create_project,
    get_project,
    save_project_report,
)

# Register each API router exactly once.
from app.project_api import router as project_router
from app.feature_api import router as feature_router
from app.prediction_api import router as prediction_router


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

app = FastAPI(
    title="Data Analyst Agent",
    description=(
        "Automated data analysis, data quality checks, insights, "
        "feature engineering, predictions, SQL, and dashboards."
    ),
    version="0.3.5",
)

app.include_router(project_router)
app.include_router(feature_router)
app.include_router(prediction_router)


# ============================================================
# 2. RESPONSE MODEL
# ============================================================

class DatasetAnalysisResponse(BaseModel):
    """Structure returned after successful dataset analysis."""

    filename: str
    file_size_bytes: int
    analysis: dict[str, Any]
    data_quality: dict[str, Any]
    insights: dict[str, Any]
    dashboard: dict[str, Any]
    notice: str
    saved_report: dict[str, Any] | None = None
    feature_engineering: dict[str, Any] | None = None


# ============================================================
# 3. JSON-SAFE VALUE CONVERSION
# ============================================================

def clean_value(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-safe Python values."""

    if value is None or value is pd.NA or value is pd.NaT:
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

    if isinstance(value, np.ndarray):
        return [clean_value(item) for item in value.tolist()]

    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except (TypeError, ValueError):
        pass

    # Convert other NumPy scalar types where possible.
    if isinstance(value, np.generic):
        return value.item()

    return value


# ============================================================
# 4. STATISTICAL ANALYSIS
# ============================================================

def analyze_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Generate descriptive statistics for a DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a Pandas DataFrame.")

    numeric_df = df.select_dtypes(include="number")

    categorical_df = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    )

    numeric_summary: dict[str, Any] = {}
    categorical_summary: dict[str, Any] = {}

    # Numerical statistics.
    for column in numeric_df.columns:
        series = pd.to_numeric(df[column], errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan).dropna()

        numeric_summary[str(column)] = {
            "count": int(series.count()),
            "mean": clean_value(series.mean()),
            "median": clean_value(series.median()),
            "minimum": clean_value(series.min()),
            "maximum": clean_value(series.max()),
            "standard_deviation": clean_value(series.std()),
        }

    # Most frequent categorical values.
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

    # Missing values by column.
    missing_values = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }

    # Column data types.
    column_types = {
        str(column): str(dtype)
        for column, dtype in df.dtypes.items()
    }

    duplicate_count = int(df.duplicated().sum())

    # Initial findings.
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
# 5. DATASET LOADING AND VALIDATION
# ============================================================

def load_dataset(filename: str, content: bytes) -> pd.DataFrame:
    """Load a validated CSV or Excel file into a DataFrame."""

    extension = Path(filename).suffix.lower()

    try:
        if extension == ".csv":
            # utf-8-sig also handles CSV files with a BOM.
            try:
                df = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(BytesIO(content), encoding="latin-1")

        elif extension == ".xlsx":
            df = pd.read_excel(
                BytesIO(content),
                engine="openpyxl",
            )

        elif extension == ".xls":
            df = pd.read_excel(
                BytesIO(content),
                engine="xlrd",
            )

        else:
            raise HTTPException(
                status_code=400,
                detail="Upload a CSV, XLSX, or XLS file.",
            )

    except HTTPException:
        raise

    except ImportError as exc:
        logger.exception("A dataset-reading dependency is missing")
        raise HTTPException(
            status_code=500,
            detail=(
                "A required file-reading dependency is missing. "
                "Install openpyxl for XLSX files and xlrd for XLS files."
            ),
        ) from exc

    except (
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError,
        OSError,
        EOFError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the uploaded file. Check that it is "
                "a valid, non-corrupted CSV or Excel dataset."
            ),
        ) from exc

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

    return df


# ============================================================
# 6. HOME ENDPOINT
# ============================================================

@app.get("/", summary="Home")
def home() -> dict[str, str]:
    """Return basic API information."""

    return {
        "message": "Welcome to Data Analyst Agent",
        "status": "running",
        "version": app.version,
        "docs": "/docs",
    }


# ============================================================
# 7. DATASET ANALYSIS ENDPOINT
# ============================================================

@app.post(
    "/api/v1/datasets/analyze",
    response_model=DatasetAnalysisResponse,
    summary="Analyze Dataset",
    description=(
        "Upload a CSV or Excel file to generate statistical analysis, "
        "data quality checks, insights, KPIs, and dashboard charts. "
        "The report is saved to a persistent project."
    ),
    responses={
        400: {"description": "Invalid or unsupported dataset."},
        404: {"description": "Selected project was not found."},
        413: {"description": "Uploaded file exceeds 25 MB."},
        500: {"description": "Dataset analysis failed."},
    },
)
async def analyze_dataset(
    file: UploadFile = File(...),
    project_id: str | None = Form(default=None),
    new_project: bool = Form(default=False),
) -> DatasetAnalysisResponse:
    """Analyze an uploaded dataset and save the report."""

    try:
        # Validate the filename and extension.
        filename = Path(file.filename or "").name
        extension = Path(filename).suffix.lower()

        if not filename or extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Upload a CSV, XLSX, or XLS file.",
            )

        # An empty project_id from a multipart form means no selection.
        project_id = (project_id or "").strip() or None

        # Validate the selected project unless a new one was requested.
        if (
            project_id is not None
            and not new_project
            and get_project(project_id) is None
        ):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Project not found. Select an existing project "
                    "or enable New Project."
                ),
            )

        # Read in chunks and enforce the upload limit.
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

        # Load and validate the dataset.
        df = load_dataset(filename, content)

        # Run analysis modules against the uploaded data.
        analysis_report = analyze_dataframe(df)
        quality_report = assess_data_quality(df)
        insights_report = generate_insights(df)
        dashboard_report = generate_dashboard(df)

        response = DatasetAnalysisResponse(
            filename=filename,
            file_size_bytes=total_bytes,
            analysis=analysis_report,
            data_quality=quality_report,
            insights=insights_report,
            dashboard=dashboard_report,
            notice=(
                "Reports describe the uploaded dataset. "
                "They do not guarantee data correctness or prediction "
                "accuracy. Review data quality warnings before making "
                "business decisions."
            ),
        )

        # Build a readable project name from the dataset filename.
        project_name = " ".join(Path(filename).stem.split())[:120]
        project_name = project_name or "Untitled Dataset"

        # Create a separate project when explicitly requested.
        if new_project:
            project = create_project(project_name)
            project_id = project["id"]

        # Reuse the selected project when supplied.
        elif project_id is not None:
            project = get_project(project_id)

            if project is None:
                raise HTTPException(
                    status_code=404,
                    detail="The selected project no longer exists.",
                )

        # Otherwise, reuse or create a project named after the dataset.
        else:
            project = get_or_create_project(project_name)
            project_id = project["id"]

        # Prepare a JSON-safe report for SQLite storage.
        report_data = jsonable_encoder(
            response,
            exclude={"saved_report"},
        )

        saved_record = save_project_report(
            project_id=project_id,
            dataset_name=filename,
            report=report_data,
        )

        if saved_record is None:
            raise HTTPException(
                status_code=404,
                detail="Project not found while saving the report.",
            )

        response.saved_report = saved_record

        return response

    except HTTPException:
        # Keep deliberate HTTP status codes and messages.
        raise

    except Exception as exc:
        logger.exception("Dataset analysis failed")

        raise HTTPException(
            status_code=500,
            detail=(
                "Dataset analysis failed. Check the server terminal "
                "for the technical error."
            ),
        ) from exc

    finally:
        # Close the upload even when processing fails.
        await file.close()
