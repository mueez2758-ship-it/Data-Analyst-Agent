
# ============================================================
# FILE: app/project_store.py
# PURPOSE: Persist projects and analysis reports in SQLite.
# ============================================================

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "projects.db"


def _connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_project_store() -> None:
    """Create project and report tables if they do not exist."""

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


def create_project(name: str) -> dict[str, Any]:
    name = " ".join(name.strip().split())

    if not name:
        raise ValueError("Project name cannot be empty.")

    if len(name) > 120:
        raise ValueError("Project name cannot exceed 120 characters.")

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
    }


def list_projects() -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute("""
            SELECT
                p.id,
                p.name,
                p.created_at,
                p.updated_at,
                COUNT(r.id) AS report_count
            FROM projects p
            LEFT JOIN project_reports r ON r.project_id = p.id
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

        reports = connection.execute(
            """
            SELECT id, dataset_name, created_at, report_json
            FROM project_reports
            WHERE project_id = ?
            ORDER BY created_at DESC
            """,
            (project_id,),
        ).fetchall()

    return {
        **dict(project),
        "reports": [
            {
                "id": report["id"],
                "dataset_name": report["dataset_name"],
                "created_at": report["created_at"],
                "report": json.loads(report["report_json"]),
            }
            for report in reports
        ],
    }


def save_project_report(
    project_id: str,
    dataset_name: str,
    report: dict[str, Any],
) -> dict[str, Any] | None:
    """Save a JSON-compatible analysis report to its project."""

    report_id = str(uuid4())
    timestamp = _now()

    try:
        report_json = json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The analysis report contains values that cannot be saved as JSON."
        ) from exc

    with _connect() as connection:
        project = connection.execute(
            "SELECT id FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()

        if project is None:
            return None

        connection.execute(
            """
            INSERT INTO project_reports
                (id, project_id, dataset_name, created_at, report_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                report_id,
                project_id,
                Path(dataset_name).name,
                timestamp,
                report_json,
            ),
        )

        connection.execute(
            "UPDATE projects SET updated_at = ? WHERE id = ?",
            (timestamp, project_id),
        )

    return {
        "id": report_id,
        "project_id": project_id,
        "dataset_name": Path(dataset_name).name,
        "created_at": timestamp,
    }


def get_project_report(
    project_id: str,
    report_id: str,
) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, project_id, dataset_name, created_at, report_json
            FROM project_reports
            WHERE project_id = ? AND id = ?
            """,
            (project_id, report_id),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "dataset_name": row["dataset_name"],
        "created_at": row["created_at"],
        "report": json.loads(row["report_json"]),
    }


initialize_project_store()
