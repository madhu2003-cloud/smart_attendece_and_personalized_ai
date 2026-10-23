"""Generate the demo attendance dataset for model development.

Produces:
  1. data/attendance_dataset.csv
     A deterministic (random_state=42) synthetic dataset of 600 students with STU-formatted IDs (STU101, STU102, etc.)
  2. Attendance records for students registered in the SQLite database (source='sample')
"""

import os
import random
import sqlite3
import datetime

import numpy as np
import pandas as pd

from database import get_db

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'attendance_dataset.csv')

DEPARTMENTS = ['CSE', 'ECE', 'IT', 'MECH', 'CIVIL', 'EEE']
YEARS = ['1st', '2nd', '3rd', '4th']
SECTIONS = ['A', 'B', 'C']
N_STUDENTS = 600
RANDOM_STATE = 42

SUBJECTS = [
    'Machine Learning & AI',
    'Database Systems',
    'Data Structures',
    'Web Technologies',
    'Python Programming'
]

FIRST_NAMES = ['Aarav', 'Priya', 'Rahul', 'Sneha', 'Vikram', 'Ananya', 'Karan', 'Riya', 'Dev', 'Isha', 'Arjun', 'Meera', 'Rohan', 'Kavya', 'Aditya', 'Pooja', 'Siddharth', 'Neha']
LAST_NAMES = ['Sharma', 'Patel', 'Verma', 'Reddy', 'Malhotra', 'Gupta', 'Joshi', 'Sen', 'Kumar', 'Singh', 'Nair', 'Shah', 'Chawla', 'Mehta', 'Rao', 'Bhasin']


def generate_dataset_csv():
    """Create the synthetic CSV deterministically with STU roll numbers."""
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    rows = []
    for i in range(N_STUDENTS):
        roll = f"STU{101 + i}"
        fn = FIRST_NAMES[i % len(FIRST_NAMES)]
        ln = LAST_NAMES[(i // len(FIRST_NAMES)) % len(LAST_NAMES)]
        name = f"{fn} {ln}"
        dept = DEPARTMENTS[rng.integers(0, len(DEPARTMENTS))]
        year = YEARS[rng.integers(0, len(YEARS))]
        section = SECTIONS[rng.integers(0, len(SECTIONS))]

        total_classes = int(rng.integers(30, 46))
        prop = float(np.clip(rng.normal(0.78, 0.13), 0.35, 0.98))

        consecutive_absences = int(rng.geometric(0.35))
        consecutive_absences = min(consecutive_absences, 14)

        last7 = float(np.clip(prop + rng.normal(0.0, 0.08), 0.0, 1.0))
        last30 = float(np.clip(prop + rng.normal(0.0, 0.05), 0.0, 1.0))

        attended_days = sum(1 for _ in range(total_classes) if rng.random() < prop)
        attendance_percentage = round(100.0 * attended_days / total_classes, 1)
        late_arrivals = max(0, int(rng.normal((1 - prop) * 14, 2.5)))

        assignment_score = float(np.clip(38 + 0.5 * attendance_percentage + rng.normal(0, 8), 0, 100))
        internal_marks = float(np.clip(35 + 0.55 * attendance_percentage + rng.normal(0, 7), 0, 100))
        previous_semester_percentage = float(np.clip(42 + 0.45 * attendance_percentage + rng.normal(0, 8), 0, 100))
        study_hours = float(np.clip(rng.normal(3.0 + 0.03 * attendance_percentage, 1.2), 0, 8))

        penalty = max(0.0, 72.0 - attendance_percentage) * 2.0
        penalty += consecutive_absences * 1.2
        penalty += max(0.0, 45.0 - last7 * 100.0) * 0.8
        risk = 1 if penalty + rng.normal(0.0, 6.0) > 12.0 else 0

        rows.append({
            'roll_number': roll,
            'name': name,
            'department': dept,
            'year': year,
            'section': section,
            'total_classes': total_classes,
            'classes_attended': attended_days,
            'classes_absent': total_classes - attended_days,
            'attendance_percentage': attendance_percentage,
            'absence_percentage': round(100.0 - attendance_percentage, 1),
            'consecutive_absences': consecutive_absences,
            'attendance_last_7_days': round(last7 * 100.0, 1),
            'attendance_last_30_days': round(last30 * 100.0, 1),
            'assignment_score': round(assignment_score, 1),
            'internal_marks': round(internal_marks, 1),
            'study_hours': round(study_hours, 1),
            'previous_semester_percentage': round(previous_semester_percentage, 1),
            'number_of_late_arrivals': late_arrivals,
            'attendance_risk': risk,
        })

    df = pd.DataFrame(rows)
    flip_idx = rng.choice(df.index, size=int(len(df) * 0.03), replace=False)
    df.loc[flip_idx, 'attendance_risk'] = 1 - df.loc[flip_idx, 'attendance_risk']

    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {len(df)} student rows with STU IDs -> {CSV_PATH}")
    return df


def seed_attendance_records(days=30, base_date=None):
    """Insert sample attendance rows per subject for students registered in DB."""
    conn = get_db()
    cur = conn.cursor()
    students = cur.execute("SELECT roll_number FROM students").fetchall()

    if not students:
        conn.close()
        return 0

    rng = np.random.default_rng(RANDOM_STATE)
    today = (datetime.date.fromisoformat(base_date) if base_date else datetime.date.today())
    class_dates = []
    d = today - datetime.timedelta(days=days * 2)
    while len(class_dates) < days:
        if d.weekday() < 5:
            class_dates.append(d.isoformat())
        d += datetime.timedelta(days=1)

    inserted = 0
    for student in students:
        roll = student['roll_number']
        prop = float(np.clip(rng.normal(0.82, 0.11), 0.40, 0.98))
        for subj in SUBJECTS:
            for date_str in class_dates[:15]:  # sample dates per subject
                if rng.random() < prop:
                    hour = rng.integers(8, 16)
                    minute = rng.integers(0, 60)
                    time_str = f"{hour:02d}:{minute:02d}:00"
                    try:
                        cur.execute("""
                            INSERT OR IGNORE INTO attendance (roll_number, subject, date, time, status, source)
                            VALUES (?, ?, ?, ?, 'Present', 'sample')
                        """, (roll, subj, date_str, time_str))
                        inserted += cur.rowcount if cur.rowcount > 0 else 0
                    except sqlite3.IntegrityError:
                        continue
    conn.commit()
    conn.close()
    print(f"Seeded {inserted} subject attendance records.")
    return inserted


if __name__ == '__main__':
    generate_dataset_csv()
    seed_attendance_records()
