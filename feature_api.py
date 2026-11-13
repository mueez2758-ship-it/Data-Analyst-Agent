
"""Feature engineering API endpoints for the Data Analyst Agent."""

from io import BytesIO
from pathlib import Path
from typing import Any
import logging

import pandas as pd
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.feature_engineering import engineer_features
from app.project_store import (
    create_project,
    get_or_create_project,
    get_project,
    save_project_report,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/datasets",
    tags=["Feature Engineering"],
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


async def _read_feature_dataset(
    file: UploadFile,
) -> tuple[str, bytes, pd.DataFrame]:
    """Validate and read a dataset without modifying its original bytes."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a CSV, XLSX, or XLS dataset.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 25 MB upload limit.",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

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
            status_code=500,
            detail=(
                "An Excel dependency is missing. Install openpyxl "
                "for XLSX files or xlrd for XLS files."
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
                "Could not read the dataset. Check its format "
                "and contents."
            ),
        ) from exc

    if not df.columns.is_unique:
        raise HTTPException(
            status_code=400,
            detail="Column names must be unique.",
        )

    if df.empty or len(df.columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain rows and columns.",
        )

    return filename, content, df


def _resolve_feature_project(
    filename: str,
    project_id: str | None,
    new_project: bool,
) -> dict[str, Any]:
    """Select an existing project or resolve a dataset-named project."""
    project_name = " ".join(Path(filename).stem.split()).strip()
    project_name = project_name[:120] or "Untitled Dataset"

    if new_project:
        project = create_project(project_name)
    elif project_id:
        project = get_project(project_id)

        if project is None:
            raise HTTPException(
                status_code=404,
                detail="Selected project was not found.",
            )
    else:
        project = get_or_create_project(project_name)

    if not isinstance(project, dict) or not project.get("id"):
        raise HTTPException(
            status_code=500,
            detail="Could not resolve a valid project.",
        )

    return project


@router.post(
    "/feature-engineer",
    summary="Engineer dataset features",
    description=(
        "Apply selected feature transformations, save the transformation "
        "summary to a project, and download the engineered dataset as CSV."
    ),
    responses={
        200: {"description": "Engineered CSV download."},
        400: {"description": "Invalid dataset or transformation options."},
        404: {"description": "Selected project was not found."},
        413: {"description": "Uploaded file exceeds 25 MB."},
        500: {"description": "Feature engineering or report saving failed."},
    },
)
async def feature_engineer_dataset(
    file: UploadFile = File(...),
    drop_duplicates: bool = Form(False),
    fill_numeric_missing: bool = Form(False),
    fill_categorical_missing: bool = Form(False),
    encode_categoricals: bool = Form(False),
    extract_date_features: bool = Form(False),
    project_id: str | None = Form(default=None),
    new_project: bool = Form(default=False),
) -> StreamingResponse:
    """Transform a dataset, save its report, and return an engineered CSV."""
    try:
        filename, original_content, df = await _read_feature_dataset(file)

        try:
            transformed, summary = engineer_features(
                df,
                drop_duplicates=drop_duplicates,
                fill_numeric_missing=fill_numeric_missing,
                fill_categorical_missing=fill_categorical_missing,
                encode_categoricals=encode_categoricals,
                extract_date_features=extract_date_features,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not engineer features: {exc}",
            ) from exc

        # Resolve the project only after transformation succeeds.
        project = _resolve_feature_project(
            filename=filename,
            project_id=project_id,
            new_project=new_project,
        )

        report_data = {
            "filename": filename,
            "report_type": "feature_engineering",
            "options": {
                "drop_duplicates": drop_duplicates,
                "fill_numeric_missing": fill_numeric_missing,
                "fill_categorical_missing": fill_categorical_missing,
                "encode_categoricals": encode_categoricals,
                "extract_date_features": extract_date_features,
            },
            "summary": summary,
            "original_file_preserved": True,
            "original_file_size_bytes": len(original_content),
        }

        try:
            saved_report = save_project_report(
                project_id=project["id"],
                dataset_name=filename,
                report=report_data,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not save the feature report: {exc}",
            ) from exc

        if saved_report is None:
            raise HTTPException(
                status_code=500,
                detail="The feature-engineering report was not saved.",
            )

        # Return the transformed data as a downloadable CSV.
        output = transformed.to_csv(index=False).encode("utf-8-sig")
        download_name = f"{Path(filename).stem}_engineered.csv"

        headers = {
            "Content-Disposition": (
                f'attachment; filename="{download_name}"'
            ),
            "X-Project-Id": str(project["id"]),
            "X-Saved-Report-Id": str(saved_report["id"]),
            "X-Feature-Engineering-Status": "completed",
        }

        return StreamingResponse(
            BytesIO(output),
            media_type="text/csv; charset=utf-8",
            headers=headers,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Feature engineering failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Feature engineering failed unexpectedly. "
                "Check the server terminal for details."
            ),
        ) from exc

    finally:
        await file.close()
