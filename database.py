"""SQLite database layer for Smart Attendance AI.

Schema aligned with project requirements:
  users       - id, role (student/faculty), username/roll_number, name, password, email, phone, subject, department, year, section
  students    - id, roll_number (STU...), name, department, year, section, email, phone, password
  attendance  - id, roll_number, subject, date, time, status, source
  notifications - id, sender_faculty, recipient_roll, student_name, subject, title, message, risk_level, is_read, created_at
  face_samples - id, roll_number, image_path, captured_at
  model_metrics - evaluation results for attendance-risk ML models
  face_model_metrics - evaluation results for face-recognition benchmark models
"""

import os
import re
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), 'attendance.db')


def format_roll_number(raw_val):
    """Normalize student roll numbers consistently to uppercase alphanumeric with STU prefix."""
    clean = re.sub(r'[^A-Za-z0-9]', '', (raw_val or '').strip().upper())
    if not clean:
        return ''
    if not clean.startswith('STU'):
        clean = 'STU' + clean
    return clean


def get_db():
    """Open a connection with row access by column name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they do not exist. Safe to call repeatedly."""
    conn = get_db()
    cur = conn.cursor()

    # Unified Users table for Authentication (Faculty & Student)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            role        TEXT NOT NULL,               -- 'student' | 'faculty'
            username    TEXT UNIQUE NOT NULL,        -- roll_number for student (STU...), name/id for faculty
            name        TEXT NOT NULL,
            password    TEXT NOT NULL,
            email       TEXT NOT NULL,
            phone       TEXT,
            subject     TEXT DEFAULT 'Machine Learning & AI', -- For faculty
            department  TEXT DEFAULT 'CSE',
            year        TEXT DEFAULT '3rd',
            section     TEXT DEFAULT 'A',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Students master registry (roll_number MUST start with STU followed by digits)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            department  TEXT NOT NULL DEFAULT 'CSE',
            year        TEXT NOT NULL DEFAULT '3rd',
            section     TEXT NOT NULL DEFAULT 'A',
            email       TEXT NOT NULL,
            phone       TEXT DEFAULT '9876543210',
            password    TEXT DEFAULT 'student123',
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Attendance log with Subject isolation
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT NOT NULL,
            subject     TEXT NOT NULL DEFAULT 'Machine Learning & AI',
            date        TEXT NOT NULL,          -- YYYY-MM-DD
            time        TEXT NOT NULL,          -- HH:MM:SS
            status      TEXT NOT NULL DEFAULT 'Present',
            source      TEXT NOT NULL DEFAULT 'manual',  -- manual | sample | face_ai
            FOREIGN KEY (roll_number) REFERENCES students(roll_number),
            UNIQUE (roll_number, subject, date)
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

    # Evaluation results for attendance-risk ML models (realistic metrics)
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

    # Evaluation results for face-recognition benchmark models (LFW-trained)
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

    # Targeted Notifications table for Faculty-to-Student Communication
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_faculty TEXT NOT NULL,
            recipient_roll TEXT NOT NULL,              -- STU101 format
            student_name   TEXT NOT NULL,
            subject        TEXT NOT NULL,
            title          TEXT NOT NULL,
            message        TEXT NOT NULL,
            risk_level     TEXT NOT NULL DEFAULT 'warning', -- 'info' | 'warning' | 'critical'
            is_read        INTEGER DEFAULT 0,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Campus Geo-Fencing configuration (Editable by Faculty ONLY)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS geofence_settings (
            id            INTEGER PRIMARY KEY DEFAULT 1,
            campus_name   TEXT NOT NULL DEFAULT 'Main Engineering Campus (Coimbatore)',
            latitude      REAL NOT NULL DEFAULT 11.000563,
            longitude     REAL NOT NULL DEFAULT 76.966575,
            radius_meters REAL NOT NULL DEFAULT 500.0,
            is_enabled    INTEGER NOT NULL DEFAULT 1,
            updated_by    TEXT DEFAULT 'Prof. Robert Vance',
            updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        INSERT OR IGNORE INTO geofence_settings (id, campus_name, latitude, longitude, radius_meters, is_enabled, updated_by)
        VALUES (1, 'Main Engineering Campus (Coimbatore)', 11.000563, 76.966575, 500.0, 1, 'Prof. Robert Vance')
    """)

    conn.commit()

    # Seed default users & students if empty
    seed_initial_data(conn)

    conn.close()


def seed_initial_data(conn):
    """Seed initial faculty accounts only. Zero inbuilt dummy student records."""
    cur = conn.cursor()
    
    # Check if faculty exists
    fac_count = cur.execute("SELECT COUNT(*) FROM users WHERE role='faculty'").fetchone()[0]
    if fac_count == 0:
        cur.execute("""
            INSERT INTO users (role, username, name, password, email, phone, subject, department)
            VALUES ('faculty', 'Prof. Vance', 'Prof. Robert Vance', 'faculty123', 'vance@university.edu', '9876500001', 'Machine Learning & AI', 'CSE')
        """)
        cur.execute("""
            INSERT INTO users (role, username, name, password, email, phone, subject, department)
            VALUES ('faculty', 'Dr. Sarah Jenkins', 'Dr. Sarah Jenkins', 'faculty123', 'sjenkins@university.edu', '9876500002', 'Database Systems', 'CSE')
        """)
        cur.execute("""
            INSERT INTO users (role, username, name, password, email, phone, subject, department)
            VALUES ('faculty', 'Prof. Alan Turing', 'Prof. Alan Turing', 'faculty123', 'aturing@university.edu', '9876500003', 'Data Structures', 'CSE')
        """)

    conn.commit()


def rebuild_db():
    """Drop every table and recreate the schema from scratch."""
    conn = get_db()
    cur = conn.cursor()
    for table in ('users', 'students', 'attendance', 'face_samples',
                  'model_metrics', 'face_model_metrics', 'notifications'):
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.close()
    init_db()


if __name__ == '__main__':
    rebuild_db()
    print("Database rebuilt with STU authentication, user roles, subject isolation, and targeted notifications.")
