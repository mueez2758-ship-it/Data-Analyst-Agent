
"""Persistent SQLite storage for projects, reports, and trained models."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "projects.db"
MODEL_DIR = DATA_DIR / "models"


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("The supplied data cannot be saved as valid JSON.") from exc


def initialize_project_store() -> None:
    """Create tables without deleting existing projects or reports."""
    with _connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS project_reports (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                report_json TEXT NOT NULL,
                FOREIGN KEY(project_id)
                    REFERENCES projects(id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS project_models (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                target_column TEXT NOT NULL,
                task_type TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                feature_columns_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(project_id)
                    REFERENCES projects(id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_project_models_project
            ON project_models(project_id, created_at DESC)
        """)


def _normalize_project_name(name: str) -> str:
    name = " ".join(name.strip().split())
    if not name:
        raise ValueError("Project name cannot be empty.")
    if len(name) > 120:
        raise ValueError("Project name cannot exceed 120 characters.")
    return name


def create_project(name: str) -> dict[str, Any]:
    name = _normalize_project_name(name)
    project_id = str(uuid4())
    timestamp = _now()

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO projects (id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (project_id, name, timestamp, timestamp),
        )

    return {
        "id": project_id,
        "name": name,
        "created_at": timestamp,
        "updated_at": timestamp,
        "report_count": 0,
        "model_count": 0,
    }


def list_projects() -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute("""
            SELECT
                p.id,
                p.name,
                p.created_at,
                p.updated_at,
                COUNT(DISTINCT r.id) AS report_count,
                COUNT(DISTINCT m.id) AS model_count
            FROM projects AS p
            LEFT JOIN project_reports AS r ON r.project_id = p.id
            LEFT JOIN project_models AS m ON m.project_id = p.id
            GROUP BY p.id
            ORDER BY p.updated_at DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_project(project_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        project = connection.execute(
            "SELECT * FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

        if project is None:
            return None

        reports = connection.execute("""
            SELECT id, dataset_name, created_at, report_json
            FROM project_reports
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,)).fetchall()

        models = connection.execute("""
            SELECT id, dataset_name, target_column, task_type,
                   algorithm, metrics_json, created_at
            FROM project_models
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,)).fetchall()

    return {
        **dict(project),
        "reports": [
            {
                "id": row["id"],
                "dataset_name": row["dataset_name"],
                "created_at": row["created_at"],
                "report": json.loads(row["report_json"]),
            }
            for row in reports
        ],
        "models": [
            {
                "id": row["id"],
                "dataset_name": row["dataset_name"],
                "target_column": row["target_column"],
                "task_type": row["task_type"],
                "algorithm": row["algorithm"],
                "metrics": json.loads(row["metrics_json"]),
                "created_at": row["created_at"],
            }
            for row in models
        ],
    }


def save_project_report(
    project_id: str,
    dataset_name: str,
    report: dict[str, Any],
) -> dict[str, Any] | None:
    report_json = _json_safe(report)
    safe_dataset_name = Path(dataset_name).name

    if not safe_dataset_name:
        raise ValueError("Dataset name cannot be empty.")

    report_id = str(uuid4())
    timestamp = _now()

    with _connect() as connection:
        project = connection.execute(
            "SELECT id FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

        if project is None:
            return None

        connection.execute("""
            INSERT INTO project_reports
                (id, project_id, dataset_name, created_at, report_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            report_id, project_id, safe_dataset_name,
            timestamp, report_json,
        ))

        connection.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (timestamp, project_id),
        )

    return {
        "id": report_id,
        "project_id": project_id,
        "dataset_name": safe_dataset_name,
        "created_at": timestamp,
    }


def get_project_report(
    project_id: str,
    report_id: str,
) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute("""
            SELECT id, project_id, dataset_name, created_at, report_json
            FROM project_reports
            WHERE project_id = ? AND id = ?
        """, (project_id, report_id)).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "dataset_name": row["dataset_name"],
        "created_at": row["created_at"],
        "report": json.loads(row["report_json"]),
    }


def save_project_model(
    project_id: str,
    dataset_name: str,
    target_column: str,
    task_type: str,
    algorithm: str,
    model: Any,
    feature_columns: list[str],
    metrics: dict[str, Any],
) -> dict[str, Any] | None:
    """Save a trusted, locally trained model pipeline and its metadata."""
    import joblib

    if not feature_columns:
        raise ValueError("At least one model feature is required.")

    if task_type not in {"classification", "regression"}:
        raise ValueError("Unsupported model task type.")

    safe_dataset_name = Path(dataset_name).name
    if not safe_dataset_name:
        raise ValueError("Dataset name cannot be empty.")

    metrics_json = _json_safe(metrics)
    features_json = _json_safe(feature_columns)

    with _connect() as connection:
        project = connection.execute(
            "SELECT id FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

        if project is None:
            return None

        model_id = str(uuid4())
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        artifact_path = MODEL_DIR / f"{model_id}.joblib"
        temporary_path = MODEL_DIR / f"{model_id}.tmp"

        try:
            joblib.dump(model, temporary_path)
            temporary_path.replace(artifact_path)

            timestamp = _now()
            connection.execute("""
                INSERT INTO project_models (
                    id, project_id, dataset_name, target_column,
                    task_type, algorithm, artifact_path,
                    feature_columns_json, metrics_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                project_id,
                safe_dataset_name,
                target_column,
                task_type,
                algorithm,
                str(artifact_path.resolve()),
                features_json,
                metrics_json,
                timestamp,
            ))

            connection.execute(
                "UPDATE projects SET updated_at = ? WHERE id = ?",
                (timestamp, project_id),
            )

        except Exception:
            artifact_path.unlink(missing_ok=True)
            temporary_path.unlink(missing_ok=True)
            raise

    return {
        "id": model_id,
        "project_id": project_id,
        "dataset_name": safe_dataset_name,
        "target_column": target_column,
        "task_type": task_type,
        "algorithm": algorithm,
        "created_at": timestamp,
    }


def list_project_models(project_id: str) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute("""
            SELECT id, project_id, dataset_name, target_column,
                   task_type, algorithm, metrics_json, created_at
            FROM project_models
            WHERE project_id = ?
            ORDER BY created_at DESC
        """, (project_id,)).fetchall()

    return [
        {
            "id": row["id"],
            "project_id": row["project_id"],
            "dataset_name": row["dataset_name"],
            "target_column": row["target_column"],
            "task_type": row["task_type"],
            "algorithm": row["algorithm"],
            "metrics": json.loads(row["metrics_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_project_model(
    project_id: str,
    model_id: str,
) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute("""
            SELECT *
            FROM project_models
            WHERE project_id = ? AND id = ?
        """, (project_id, model_id)).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "dataset_name": row["dataset_name"],
        "target_column": row["target_column"],
        "task_type": row["task_type"],
        "algorithm": row["algorithm"],
        "artifact_path": row["artifact_path"],
        "feature_columns": json.loads(row["feature_columns_json"]),
        "metrics": json.loads(row["metrics_json"]),
        "created_at": row["created_at"],
    }


def load_project_model(project_id: str, model_id: str) -> tuple[Any, dict[str, Any]] | None:
    """Load only a model artifact registered to the requested project."""
    import joblib

    metadata = get_project_model(project_id, model_id)
    if metadata is None:
        return None

    artifact_path = Path(metadata["artifact_path"]).resolve()
    model_root = MODEL_DIR.resolve()

    if artifact_path.parent != model_root or artifact_path.suffix != ".joblib":
        raise ValueError("Invalid model artifact location.")

    if not artifact_path.is_file():
        raise FileNotFoundError("The saved model artifact is missing.")

    # Only load artifacts generated and stored by this application.
    model = joblib.load(artifact_path)
    return model, metadata


def get_or_create_project(name: str) -> dict[str, Any]:
    name = " ".join(name.strip().split()) or "Untitled Dataset"
    name = name[:120].strip() or "Untitled Dataset"

    with _connect() as connection:
        row = connection.execute("""
            SELECT
                p.id, p.name, p.created_at, p.updated_at,
                COUNT(DISTINCT r.id) AS report_count,
                COUNT(DISTINCT m.id) AS model_count
            FROM projects AS p
            LEFT JOIN project_reports AS r ON r.project_id = p.id
            LEFT JOIN project_models AS m ON m.project_id = p.id
            WHERE p.name = ? COLLATE NOCASE
            GROUP BY p.id
            ORDER BY p.updated_at DESC
            LIMIT 1
        """, (name,)).fetchone()

    if row is not None:
        return dict(row)

    return create_project(name)


initialize_project_store()
