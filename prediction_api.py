
"""FastAPI endpoints for Automated ML Studio.

Features:
- Recommend likely prediction target columns.
- Train and evaluate prediction models.
- Save evaluation reports and trained models.
- Reuse an existing project or create a new project.
- List saved models in a project.
- Predict new datasets using previously saved models.
"""

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder

from app.predictions import (
    recommend_target_columns,
    train_and_evaluate,
)

from app.project_store import (
    create_project,
    get_or_create_project,
    get_project,
    list_project_models,
    load_project_model,
    save_project_model,
    save_project_report,
)


router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["Automated ML Studio"],
)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


async def _read_dataset(
    file: UploadFile,
) -> tuple[str, pd.DataFrame]:
    """Validate and read an uploaded CSV or Excel dataset."""

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
            status_code=400,
            detail=(
                "Missing Excel dependency. Install openpyxl "
                "for XLSX or xlrd for XLS files."
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

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="The uploaded dataset contains no rows.",
        )

    if len(df.columns) < 1:
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain at least one column.",
        )

    if not df.columns.is_unique:
        raise HTTPException(
            status_code=400,
            detail="Column names must be unique.",
        )

    return filename, df


def _dataset_project_name(filename: str) -> str:
    """Generate a readable project name from the dataset filename."""

    name = " ".join(Path(filename).stem.split()).strip()

    return name[:120] or "Untitled Dataset"


def _resolve_project(
    filename: str,
    project_id: str | None,
    new_project: bool,
) -> dict[str, Any]:
    """Find an existing project or create a new dataset project."""

    if new_project and project_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Choose either an existing project or a new project, "
                "not both."
            ),
        )

    if new_project:
        project = create_project(
            _dataset_project_name(filename)
        )

    elif project_id:
        project = get_project(project_id)

        if project is None:
            raise HTTPException(
                status_code=404,
                detail="Selected project was not found.",
            )

    else:
        project = get_or_create_project(
            _dataset_project_name(filename)
        )

    if not isinstance(project, dict):
        raise HTTPException(
            status_code=500,
            detail="The project store returned an invalid project.",
        )

    return project


def _prepare_inference_features(
    df: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Prepare new data using the saved pipeline's expected columns."""

    working = df.copy()
    working = working.replace([np.inf, -np.inf], np.nan)

    # Recreate date components if the training pipeline used them.
    for feature in feature_columns:
        if feature in working.columns:
            continue

        for suffix, component in (
            ("_year", "year"),
            ("_month", "month"),
            ("_day", "day"),
        ):
            if feature.endswith(suffix):
                original_column = feature[:-len(suffix)]

                if original_column in working.columns:
                    dates = pd.to_datetime(
                        working[original_column],
                        errors="coerce",
                    )

                    working[feature] = getattr(dates.dt, component)

                break

    missing_columns = [
        column
        for column in feature_columns
        if column not in working.columns
    ]

    if missing_columns:
        raise ValueError(
            "The uploaded dataset is missing required model features: "
            + ", ".join(missing_columns)
        )

    # Keep the original training feature order.
    X_new = working.loc[:, feature_columns].copy()

    # Read the numeric and categorical feature groups from
    # the fitted ColumnTransformer.
    preprocessor = model.named_steps["preprocessor"]

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue

        if name == "numeric":
            numeric_columns = list(columns)

        elif name == "categorical":
            categorical_columns = list(columns)

    # Apply the same basic type normalization used during training.
    for column in numeric_columns:
        X_new[column] = pd.to_numeric(
            X_new[column],
            errors="coerce",
        )

    for column in categorical_columns:
        X_new[column] = (
            X_new[column].astype("string").astype(object)
        )

    return X_new


@router.post(
    "/recommend-targets",
    summary="Recommend likely prediction target columns",
)
async def recommend_targets(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Recommend columns that may be suitable prediction targets."""

    try:
        filename, df = await _read_dataset(file)

        try:
            recommendations = recommend_target_columns(df)

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        return {
            "filename": filename,
            "target_recommendations": jsonable_encoder(
                recommendations
            ),
            "notice": (
                "Recommendations are heuristic suggestions. "
                "They do not prove that a column is a suitable "
                "prediction target."
            ),
        }

    finally:
        await file.close()


@router.post(
    "/evaluate",
    summary="Train, evaluate, and save a prediction model",
)
async def evaluate_prediction_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form("auto"),
    test_size: float = Form(0.2),
    project_id: str | None = Form(default=None),
    new_project: bool = Form(default=False),
) -> dict[str, Any]:
    """Train candidate models, select the best, and save the model."""

    try:
        filename, df = await _read_dataset(file)

        normalized_target = target_column.strip()
        normalized_task = task_type.strip().lower()

        if not normalized_target:
            raise HTTPException(
                status_code=400,
                detail="Target column cannot be empty.",
            )

        # "infer" is accepted as an alias for automatic task detection.
        if normalized_task == "infer":
            normalized_task = "auto"

        if normalized_task not in {
            "auto",
            "classification",
            "regression",
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid task_type. Choose auto, "
                    "classification, or regression."
                ),
            )

        if not 0.1 <= test_size <= 0.4:
            raise HTTPException(
                status_code=400,
                detail="test_size must be between 0.1 and 0.4.",
            )

        if normalized_target not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{normalized_target}' "
                    "does not exist in the dataset."
                ),
            )

        # Train and return both the report and fitted pipeline.
        try:
            result, fitted_model, feature_columns = train_and_evaluate(
                df=df,
                target_column=normalized_target,
                task_type=normalized_task,
                test_size=test_size,
                return_model=True,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Model training or evaluation failed.",
            ) from exc

        # Resolve the project after successful model evaluation.
        project = _resolve_project(
            filename=filename,
            project_id=project_id,
            new_project=new_project,
        )

        resolved_project_id = project.get("id")

        if not resolved_project_id:
            raise HTTPException(
                status_code=500,
                detail="The project store did not return a project ID.",
            )

        prediction_data = jsonable_encoder(result)

        actual_task = result["task"]
        selected_algorithm = result["selected_model"]
        test_metrics = result["test_metrics"]

        # Save the fitted pipeline and its metadata.
        try:
            saved_model = save_project_model(
                project_id=resolved_project_id,
                dataset_name=filename,
                target_column=normalized_target,
                task_type=actual_task,
                algorithm=selected_algorithm,
                model=fitted_model,
                feature_columns=feature_columns,
                metrics=test_metrics,
            )

        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not save the trained model: {exc}",
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="An error occurred while saving the trained model.",
            ) from exc

        if saved_model is None:
            raise HTTPException(
                status_code=404,
                detail="The selected project was not found.",
            )

        report_data = {
            "filename": filename,
            "report_type": "prediction_evaluation",
            "target_column": normalized_target,
            "task_type": actual_task,
            "test_size": test_size,
            "prediction": prediction_data,
            "saved_model_id": saved_model["id"],
        }

        # Persist the complete evaluation report.
        try:
            saved_report = save_project_report(
                project_id=resolved_project_id,
                dataset_name=filename,
                report=report_data,
            )

        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail="Could not serialize the prediction report.",
            ) from exc

        if saved_report is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "The prediction report was not saved. "
                    "The trained model may already have been saved."
                ),
            )

        return {
            "filename": filename,
            "project": jsonable_encoder(project),
            "prediction": prediction_data,
            "saved_model": jsonable_encoder(saved_model),
            "saved_report": jsonable_encoder(saved_report),
            "notice": (
                "The trained model and evaluation report were saved. "
                "Metrics estimate performance on the held-out test split "
                "and do not guarantee future performance."
            ),
        }

    finally:
        await file.close()


@router.get(
    "/projects/{project_id}/models",
    summary="List trained models saved in a project",
)
async def get_saved_project_models(
    project_id: str,
) -> dict[str, Any]:
    """Return model metadata for a selected project."""

    project = get_project(project_id)

    if project is None:
        raise HTTPException(
            status_code=404,
            detail="Project was not found.",
        )

    models = list_project_models(project_id)

    return {
        "project_id": project_id,
        "project_name": project["name"],
        "model_count": len(models),
        "models": jsonable_encoder(models),
    }


@router.post(
    "/projects/{project_id}/models/{model_id}/predict",
    summary="Predict new data using a previously saved model",
)
async def predict_with_saved_model(
    project_id: str,
    model_id: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Use a saved fitted pipeline to predict a newly uploaded dataset."""

    try:
        filename, df = await _read_dataset(file)

        project = get_project(project_id)

        if project is None:
            raise HTTPException(
                status_code=404,
                detail="Project was not found.",
            )

        try:
            loaded = load_project_model(
                project_id=project_id,
                model_id=model_id,
            )

        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="The saved model artifact could not be found.",
            ) from exc

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Could not load the saved model.",
            ) from exc

        if loaded is None:
            raise HTTPException(
                status_code=404,
                detail="The selected model was not found in this project.",
            )

        fitted_model, metadata = loaded

        try:
            X_new = _prepare_inference_features(
                df=df,
                model=fitted_model,
                feature_columns=metadata["feature_columns"],
            )

            predictions = fitted_model.predict(X_new)

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed for the uploaded dataset.",
            ) from exc

        prediction_values = [
            value.item() if isinstance(value, np.generic) else value
            for value in predictions.tolist()
        ]

        return {
            "filename": filename,
            "project_id": project_id,
            "project_name": project["name"],
            "model_id": model_id,
            "algorithm": metadata["algorithm"],
            "target_column": metadata["target_column"],
            "task_type": metadata["task_type"],
            "rows_predicted": len(prediction_values),
            "predictions": jsonable_encoder(prediction_values),
            "notice": (
                "Predictions were generated using the previously saved "
                "model. No model retraining was performed."
            ),
        }

    finally:
        await file.close()
