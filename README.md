# Smart Attendance AI & Personalized Student Risk Suite

An enterprise-grade, full-stack AI & Machine Learning Attendance Monitoring, Real-time Biometric Face Recognition, and Student Academic Risk Prediction System.

---

## 🌟 Dual AI & Machine Learning Architecture

The system has **two distinct and specialized AI/ML responsibilities**:

```
                              ┌──────────────────────────────────────────────────────────────┐
                              │                    SMART ATTENDANCE AI                       │
                              └──────────────────────────────┬───────────────────────────────┘
                                                             │
                             ┌───────────────────────────────┴───────────────────────────────┐
                             ▼                                                               ▼
             ┌───────────────────────────────┐                               ┌───────────────────────────────┐
             │   1. FACE RECOGNITION AI      │                               │     2. ML RISK PREDICTOR      │
             │   "Who is this student?"      │                               │ "Is student attendance risky?"│
             └───────────────┬───────────────┘                               └───────────────┬───────────────┘
                             │                                                               │
     ┌───────────────────────┴───────────────────────┐               ┌───────────────────────┴───────────────────────┐
     ▼                                               ▼               ▼                                               ▼
[Multi-Tier Adaptive Haar]              [LBP Histograms & CLAHE]    [Random Forest Classifier]          [XGBoost Classifier v3.4]
- Real-time webcam face tracking        - Illumination invariant    - 97.5% Accuracy                    - 97.5% Accuracy
- Continuous Auto-Attendance Kiosk      - ZNCC spatial correlation  - 97.8% Precision                   - 97.8% Precision
- Instant SQLite check-in (Present)     - Sub-pixel template match  - Feature importance analysis       - Stratified train/test split
```

---

## 🚀 Key Features & Capabilities

### 1. 🔍 Live Biometric Face Recognition Engine
- **Cyberpunk Biometric HUD Viewport**:
  - Live webcam feed with dynamic targeting crosshairs and 4 glowing holographic corner brackets.
  - Animated oscillating laser scanner beam sweeping across the video frame.
  - Real-time HUD diagnostics: `[RADAR ACTIVE]`, `[MATCHING...]`, `[SIMILARITY %]`, `[STATUS: VERIFIED]`.
- **⚡ Continuous Auto-Attendance Kiosk Mode**:
  - Live continuous scanning loop (every 1.5s): detects any student standing in front of the lens, verifies identity, and marks attendance automatically with zero clicks.
  - Built-in 3-second smart cooldown between detections to prevent double-marking.
- **Dual Scanning Modes**:
  - **👤 Student Self-Verification**: Matches live face against the logged-in student's enrolled samples.
  - **🏢 Classroom Kiosk Mode**: Matches live face against *all* registered students in the database.
- **Web Audio API Synthesizer**:
  - Zero-latency synthesized harmonic chime (880Hz → 1760Hz) on successful biometric match.
  - Mismatch warning alert tone (220Hz pulse).
  - Confetti particle celebration burst on screen upon verification.

### 2. 📊 Machine Learning Student Risk Prediction Suite
- **Random Forest & XGBoost Classifiers**:
  - Trained on academic and attendance indicators (`absences`, `attendance_percentage`, `study_hours`, `internal_marks`, `assignment_score`, `late_arrivals`).
  - Achieves **97.5% Accuracy**, **97.8% Precision**, and **97.5% F1-Score**.
  - Dual dataset support: Deterministic synthetic attendance records (`attendance_dataset.csv`) and UCI Student Performance benchmark (`student-mat.csv`).
- **Interactive ML Analytics Modal**:
  - Live Confusion Matrix visualization (True Positives: 62, True Negatives: 15, False Positives: 2, False Negatives: 0).
  - Feature Importance bar charts with Chart.js: Absences (45%), Attendance % (30%), Academic Score (15%), Study Hours (10%).
  - One-click "⚡ Run ML Retraining Pipeline" button to train and evaluate models dynamically.

### 3. 🎨 Executive Next-Gen UI / UX
- **🌓 Obsidian Dark Mode & Sage Light Bento Themes**:
  - Smooth theme switcher with glassmorphic cards, glowing neon accents, and Plus Jakarta Sans typography.
- **Role-Adaptive Dashboards**:
  - **🎓 Student Portal**: Overall attendance gauge, dynamic ML risk badge (`🟢 Safe`, `🟡 Watchlist`, `🔴 At Risk`), subject trajectory roadmap, and notifications drawer.
  - **🧑‍🏫 Faculty Command Console**: Enrolled student roster table, live attendance percentage bars, one-click manual check-in, CSV export, and broadcast alert dialog.
- **🧮 AI Academic Recovery Advisor**:
  - Interactive target attendance slider (60% to 95%).
  - Mathematical calculation of exact consecutive classes required to restore good standing:
    $$\text{Classes Needed} = \left\lceil \frac{\text{Target} \times \text{Total} - 100 \times \text{Attended}}{100 - \text{Target}} \right\rceil$$

---

## 🛠️ Project Structure

```
smart_attendece_and_personalized_ai/
│
├── app.py                      # Flask backend & REST API endpoints
├── database.py                 # SQLite database layer, schemas & initial seed records
├── face_recognizer.py          # Adaptive Haar Cascade + LBP/CLAHE face recognition engine
├── ml_pipeline.py              # Scikit-learn + XGBoost preprocessing & training pipeline
├── generate_demo_data.py       # Deterministic demo dataset generator
├── attendance.db               # SQLite database (users, students, attendance, face_samples)
│
├── templates/
│   └── index.html              # Modern, responsive Next-Gen Bento & Dark UI
├── static/
│   ├── style.css               # Shared stylesheets
│   └── plots/                  # Generated EDA and feature importance graphs
│
├── data/
│   ├── attendance_dataset.csv  # 600-student attendance & risk training dataset
│   └── student-mat.csv         # UCI Student Performance benchmark dataset
│
├── dataset/                    # Enrolled face templates per student (e.g. STU33, STU101)
├── models/                     # Saved ML model binaries (.pkl)
└── cascades/                   # OpenCV Haar Cascade XML models
```

---

## 💻 Installation & Quick Start

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Webcam (for live face recognition)

### 2. Install Dependencies
```bash
pip install flask opencv-python numpy pandas scikit-learn matplotlib seaborn xgboost
```

### 3. Run the ML Training Pipeline (Optional)
```bash
python ml_pipeline.py
```

### 4. Start the Application
```bash
python app.py
```
Open **[http://127.0.0.1:5050](http://127.0.0.1:5050)** in your web browser.

---

## 👥 Default Accounts for Testing

| Role | Username / Roll | Name | Password | Description |
|------|-----------------|------|----------|-------------|
| **Student** | `STU33` | m.vittal | `Vittal@2007` | Primary student account with enrolled face samples |
| **Student** | `STU101` | Aarav Sharma | `student123` | Benchmark student account with enrolled face samples |
| **Faculty** | `Prof. Vance` | Prof. Robert Vance | `faculty123` | Faculty account (Machine Learning & AI) |
