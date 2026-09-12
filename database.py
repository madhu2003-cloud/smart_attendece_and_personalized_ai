"""SQLite database layer for Smart Attendance AI.

Schema follows the project specification:

  students   - roll_number, name, department, year, section, email
  attendance - roll_number, date, time, status (one record per student per day)
  face_samples - face images collected per student (webcam capture module)
  model_metrics - evaluation results for the attendance-risk ML models (RF / XGBoost)
  face_model_metrics - evaluation results for the face-recognition benchmark models
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), 'attendance.db')


def get_db():
    """Open a connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they do not exist. Safe to call repeatedly."""
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            department  TEXT NOT NULL,
            year        TEXT NOT NULL,
            section     TEXT NOT NULL,
            email       TEXT NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # One attendance row per student per day. The UNIQUE(roll_number, date)
    # constraint enforces "prevent the same student from being marked present
    # multiple times on the same day" at the database level.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT NOT NULL,
            date        TEXT NOT NULL,          -- YYYY-MM-DD
            time        TEXT NOT NULL,          -- HH:MM:SS
            status      TEXT NOT NULL DEFAULT 'Present',
            source      TEXT NOT NULL DEFAULT 'manual',  -- manual | sample
            FOREIGN KEY (roll_number) REFERENCES students(roll_number),
            UNIQUE (roll_number, date)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT NOT NULL,
            image_path  TEXT NOT NULL,
            captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (roll_number) REFERENCES students(roll_number)
        )
    """)

    # Evaluation results for the attendance-risk ML models.
    # confusion_matrix / classification_report / feature_importance are JSON strings.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name           TEXT UNIQUE NOT NULL,
            accuracy             REAL,
            precision_score      REAL,
            recall_score         REAL,
            recall_at_risk       REAL,
            f1_score             REAL,
            confusion_matrix     TEXT,
            classification_report TEXT,
            feature_importance   TEXT,
            trained_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Evaluation results for the face-recognition benchmark models (LFW-trained).
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_model_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT UNIQUE NOT NULL,
            accuracy        REAL NOT NULL,
            precision_score REAL NOT NULL,
            recall_score    REAL NOT NULL,
            f1_score        REAL NOT NULL,
            n_classes       INTEGER NOT NULL,
            trained_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Notifications table for Faculty and Students
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            role         TEXT NOT NULL,               -- 'faculty' | 'student'
            roll_number  TEXT,                        -- Student roll number (if student notification)
            student_name TEXT,                        -- Student name
            subject      TEXT DEFAULT 'General',      -- Subject affected
            title        TEXT NOT NULL,
            message      TEXT NOT NULL,
            risk_level   TEXT NOT NULL DEFAULT 'warning', -- 'info' | 'warning' | 'critical'
            is_read      INTEGER DEFAULT 0,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Attempt to alter existing tables for extra columns if they don't exist
    try:
        cur.execute("ALTER TABLE students ADD COLUMN password TEXT DEFAULT '123456'")
    except Exception:
        pass

    try:
        cur.execute("ALTER TABLE attendance ADD COLUMN subject TEXT DEFAULT 'General'")
    except Exception:
        pass

    conn.commit()
    conn.close()


def rebuild_db():
    """Drop every table and recreate the schema from scratch.

    Used to wipe the previous demo schema and start clean.
    """
    conn = get_db()
    cur = conn.cursor()
    for table in ('students', 'attendance', 'face_samples',
                  'model_metrics', 'face_model_metrics', 'notifications'):
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.close()
    init_db()


if __name__ == '__main__':
    rebuild_db()
    print("Database rebuilt with the spec-aligned schema.")
