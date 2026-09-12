# Smart Attendance AI

AI-based attendance monitoring, analytics and student risk prediction system.

The project has **two separate AI/ML responsibilities**, as required by the spec:

1. **Face recognition** -> "who is this person?" (face detection, face-sample
   collection, and benchmark identity recognition).
2. **Machine learning prediction** -> "is this student's attendance pattern
   putting them at risk?" (Random Forest + XGBoost trained on
   attendance/academic features).

The ML risk models are **never trained on face images**, and the face-recognition
models are **not** described as attendance-prediction models.

---

## Status: MID-REVIEW scope (implemented)

Everything below is implemented and working. Anything not listed here is
**Future Work / Post-Mid-Review** (see the table at the bottom).

| # | Component | Where |
|---|-----------|-------|
| 1 | Project structure | this folder |
| 2 | Flask backend | `app.py` |
| 3 | Student registration frontend | `templates/index.html` |
| 4 | SQLite student database | `database.py` -> `attendance.db` |
| 5 | Student record insertion | `POST /api/students` |
| 6 | Student record viewing | `GET /api/students` |
| 7 | Validation (roll format, name, email, duplicates) | `app.py` -> `validate_student` |
| 8 | Attendance table with same-day duplicate prevention | `UNIQUE(roll_number, date)` |
| 9 | Sample attendance dataset | `generate_demo_data.py` + `POST /api/attendance/seed` |
| 10 | Pandas data loading | `ml_pipeline.py` -> `load_dataset` |
| 11-15 | Cleaning, missing values, duplicates, encoding, feature engineering | `ml_pipeline.py` |
| 16-18 | EDA + visualizations + correlation heatmap | `static/plots/*.png` |
| 19 | Stratified train-test split | `ml_pipeline.py` |
| 20-21 | Random Forest + XGBoost | `ml_pipeline.py` |
| 22-25 | Evaluation, confusion matrices, feature importance, comparison | `model_metrics` table + Analytics tab |

### Modules

- **Module 1 - Student registration**: `Students` tab. Fields: roll number,
  name, department, year, section, email. Duplicate roll numbers are rejected.
- **Module 2 - Face dataset collection**: `Face Collection` tab. Webcam frames
  are checked for a face (Haar cascade) and saved to
  `dataset/<roll_number>/`, logged in the `face_samples` table.
- **Module 3 - Attendance data**: `Attendance` tab. Mark a student present
  (once per day, enforced by the database), view the log, or generate clearly
  labeled sample records (`source='sample'`) for the demo.
- **Module 4-15 - ML pipeline**: `Analytics & Models` tab. Button runs the
  whole pipeline: preprocessing -> feature engineering -> EDA plots ->
  one-hot encoding -> stratified split -> train Random Forest + XGBoost ->
  evaluate -> feature importance.

---

## How to run

```bash
# 1. Install dependencies (Python 3.10+; tested on 3.14)
pip install flask opencv-python numpy pandas scikit-learn matplotlib seaborn xgboost

# 2. (Optional) generate the synthetic demo dataset + sample attendance
python generate_demo_data.py

# 3. (Optional) train the face-recognition benchmark models on LFW
#    downloads the LFW dataset from the internet on first run
python model_trainer.py

# 4. Train the attendance-risk ML models from the command line (optional;
#    there is also a "Run Full ML Pipeline" button in the web UI)
python ml_pipeline.py

# 5. Start the web app
python app.py
# open http://127.0.0.1:5000
```

---

## Datasets - honesty notes

- **`data/attendance_dataset.csv`** is a **synthetic demo dataset** generated
  deterministically (`random_state=42`) by `generate_demo_data.py`: 600
  students, the attendance/academic features from the spec, and a binary
  `attendance_risk` target (0 = Normal, 1 = At Risk) computed from a
  documented latent-score rule plus ~3% label noise. Names are obviously
  synthetic (`Student 101`). It is for **model development only** and makes no
  real-world claims.
- Replacing it with real records is a one-file change: export attendance from
  `attendance.db` to the same columns, or edit `CSV_PATH` in `ml_pipeline.py`.
  This is planned as post-mid-review work once the system records real
  attendance.
- **Face models** (`models/pca_model.pkl`, `svm_model.pkl`, `knn_model.pkl`)
  are trained on the public **LFW benchmark** and recognize LFW identities.
  They are a demonstration of the face-recognition approach only - the UI
  labels them as such. Recognizing registered students requires training on
  the collected `dataset/<roll_number>/` samples (future work).

---

## Model results (on the demo dataset, after `python ml_pipeline.py`)

Two tree-based models are compared honestly - no claim that XGBoost is
"always better". The classes are imbalanced, so **recall on the At-Risk class**
matters more than raw accuracy (missing an at-risk student is costlier than
flagging a normal one).

| Model | Accuracy | Precision | Recall | F1 | Recall (At-Risk) |
|-------|----------|-----------|--------|-----|------------------|
| Random Forest | ~0.88 | ~0.88 | ~0.88 | ~0.88 | ~0.82 |
| XGBoost | ~0.88 | ~0.87 | ~0.88 | ~0.87 | ~0.79 |

- **Standardization is intentionally skipped**: both models are tree-based and
  do not require feature scaling. `StandardScaler` is only relevant for
  distance-based / linear models (not used here).
- **Correlation is not causation**: the heatmap shows associations (e.g.
  attendance vs marks ~ +0.75 in the demo data) but the README and UI do not
  claim one causes the other.
- Interpretation of feature importance is based on what the *trained models
  actually output* (top features on the demo data: `attendance_percentage`,
  then the recent-attendance / absence features), not on universal claims.

---

## Post-mid-review / Future Work (not implemented)

Per the spec, these are deliberately left for after the mid review and are
**not** presented as complete:

- Train face recognition on the collected student samples and integrate it
  with automatic attendance marking (recognize -> record date/time
  automatically).
- Connect real system-generated attendance data to the ML pipeline (replace
  the synthetic CSV).
- Attendance dashboard, low-attendance alerts, personalized recommendations,
  attendance reports.
- Deep-learning face recognition, deployment, mobile / multi-camera support,
  advanced authentication / notifications.

---

## Project structure

```
smartAttendanceAI/
├── app.py                 # Flask backend (all API endpoints)
├── database.py            # SQLite schema
├── generate_demo_data.py  # synthetic demo dataset + sample attendance seeding
├── ml_pipeline.py         # preprocessing, EDA, RF + XGBoost training/evaluation
├── model_trainer.py       # LFW benchmark face-model training (optional)
├── face_recognizer.py     # Haar face detection, sample collection, benchmark scan
├── cascades/              # haarcascade_frontalface_default.xml (vendored)
├── data/                  # attendance_dataset.csv (synthetic demo data)
├── dataset/               # collected face samples per roll number
├── models/                # trained model pickles
├── static/                # style.css + generated plots/
├── templates/index.html   # single-page UI
└── attendance.db          # SQLite database
```
