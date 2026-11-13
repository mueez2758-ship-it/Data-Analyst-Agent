
"""FastAPI endpoint for machine-learning evaluation."""

from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.predictions import train_and_evaluate

router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["Predictions"],
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@router.post(
    "/evaluate",
    summary="Train and evaluate a prediction model",
)
async def evaluate_prediction_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form("auto"),
    test_size: float = Form(0.2),
):
    """Evaluate a regression or classification model on held-out data."""

    try:
        # Validate the uploaded filename and extension.
        filename = Path(file.filename or "").name
        extension = Path(filename).suffix.lower()

        if not filename or extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Upload a CSV, XLSX, or XLS dataset.",
            )

        # Read at most 25 MB plus one byte to detect oversized files.
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

        # Load the uploaded dataset according to its file extension.
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
                    "Missing Excel dependency. "
                    "Install openpyxl or xlrd."
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
                    "Could not read the dataset. "
                    "Check its file format and contents."
                ),
            ) from exc

        # Check that the dataset has enough structure for analysis.
        if df.empty or len(df.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Dataset must contain rows "
                    "and at least two columns."
                ),
            )

        if not df.columns.is_unique:
            raise HTTPException(
                status_code=400,
                detail="Column names must be unique.",
            )

        # Train and evaluate the appropriate prediction model.
        try:
            result = train_and_evaluate(
                df=df,
                target_column=target_column,
                task_type=task_type.strip().lower(),
                test_size=test_size,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        # Return the filename and model evaluation results.
        return {
            "filename": filename,
            "prediction": result,
        }

    finally:
        # Always close the uploaded file, including when an error occurs.
        await file.close()
