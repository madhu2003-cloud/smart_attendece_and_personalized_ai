"""Attendance-risk ML pipeline: preprocessing, EDA, visualization, and
Random Forest + XGBoost training/evaluation.

This is the second, separate ML responsibility of the project (the first is
face recognition). Per the spec, the ML models are NEVER trained on face
images - they predict whether a student's attendance pattern puts them at risk
(attendance_risk: 0 = Normal, 1 = At Risk).

Data source: data/attendance_dataset.csv, a clearly documented synthetic demo
dataset (see generate_demo_data.py). Real attendance records collected by the
system can replace it later; the pipeline reads a plain CSV, so swapping the
file is the only change required.

Run from the project folder:
    python ml_pipeline.py            # full pipeline, prints every step
"""

import os
import json
import pickle

import warnings
import numpy as np
# Suppress FutureWarning about np.long/np.ulong removed in NumPy 2.x
warnings.filterwarnings("ignore", category=FutureWarning, message=".*np\\.long.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*np\\.ulong.*")
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless backend; no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

from database import get_db

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'attendance_dataset.csv')
PLOT_DIR = os.path.join(BASE_DIR, 'static', 'plots')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = 'attendance_risk'
IDENTIFIER_COLS = ['roll_number', 'name']
NUMERIC_FEATURES = [
    'total_classes',
    'classes_attended',
    'classes_absent',
    'attendance_percentage',
    'absence_percentage',
    'consecutive_absences',
    'attendance_last_7_days',
    'attendance_last_30_days',
    'assignment_score',
    'internal_marks',
    'study_hours',
    'previous_semester_percentage',
    'number_of_late_arrivals',
]
CATEGORICAL_FEATURES = ['department', 'year', 'section']
# Features added during feature engineering (step documented below).
ENGINEERED_FEATURES = ['attendance_gap', 'avg_academic_score']

RF_PATH = os.path.join(MODEL_DIR, 'attendance_rf.pkl')
XGB_PATH = os.path.join(MODEL_DIR, 'attendance_xgb.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'attendance_feature_columns.pkl')


# --------------------------------------------------------------------------
# Preprocessing steps (each one returns a short human-readable summary)
# --------------------------------------------------------------------------

UCI_CSV_PATH = os.path.join(DATA_DIR, 'student-mat.csv')

def load_dataset():
    """Step 1-3: load CSV (UCI Student Performance dataset from Slide 6 if present, or fallback to attendance_dataset.csv)."""
    if os.path.exists(UCI_CSV_PATH):
        df = pd.read_csv(UCI_CSV_PATH)
        # Process UCI Student Performance dataset (Slide 6 & 10)
        if 'attendance_risk' not in df.columns:
            abs_median = df['absences'].median()
            g3_median = df['G3'].median()
            df['attendance_risk'] = ((df['absences'] > abs_median) & (df['G3'] < g3_median)).astype(int)
        df['roll_number'] = [f"UCI-{100+i}" for i in range(len(df))]
        df['name'] = [f"Student {i+1}" for i in range(len(df))]
        return df
    df = pd.read_csv(CSV_PATH)
    df['roll_number'] = df['roll_number'].astype(str)
    return df


def check_shape_and_types(df):
    """Step 4-5: data types and missing values."""
    lines = [f"Shape: {df.shape[0]} rows x {df.shape[1]} columns"]
    lines.append("Data types:\n" + df.dtypes.to_string())
    missing = df.isnull().sum()
    lines.append("\nMissing values per column:\n" + missing.to_string())
    return "\n".join(lines)


def clean(df):
    """Step 6-8: handle missing values, duplicates, invalid values."""
    notes = []
    n_before = len(df)

    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        notes.append(f"Filled {df.isnull().sum().sum()} missing values in {missing_cols}.")
        for col in missing_cols:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    else:
        notes.append("No missing values found - data clean.")

    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        notes.append(f"Removed {n_dup} duplicate rows.")
    else:
        notes.append("No duplicate rows found.")

    notes.append(f"Rows before cleaning: {n_before}, after: {len(df)}")
    return df, "\n".join(notes)


def engineer_features(df):
    """Step: feature engineering (Slide 7 composite metrics).
    - Study Efficiency = (G1 + G2 + G3) / (studytime + 1)
    - Total Support = sum of schoolsup + famsup + paid flags
    - Attendance Gap / Avg Academic Score (fallback columns)
    """
    if 'G1' in df.columns and 'G2' in df.columns and 'G3' in df.columns and 'studytime' in df.columns:
        df['study_efficiency'] = (df['G1'] + df['G2'] + df['G3']) / (df['studytime'] + 1)
    if 'schoolsup' in df.columns and 'famsup' in df.columns and 'paid' in df.columns:
        df['total_support'] = ((df['schoolsup'] == 'yes').astype(int) + 
                               (df['famsup'] == 'yes').astype(int) + 
                               (df['paid'] == 'yes').astype(int))
    if 'attendance_percentage' in df.columns and 'previous_semester_percentage' in df.columns:
        df['attendance_gap'] = df['attendance_percentage'] - df['previous_semester_percentage']
    if 'assignment_score' in df.columns and 'internal_marks' in df.columns:
        df['avg_academic_score'] = df[['assignment_score', 'internal_marks']].mean(axis=1)
    return df


def encode_categoricals(df):
    """Step 9-10: encode categorical variables for the models."""
    obj_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    obj_cols = [c for c in obj_cols if c not in IDENTIFIER_COLS]
    df = pd.get_dummies(df, columns=obj_cols, dtype=int)
    return df


def check_class_distribution(y):
    """Step 12: class distribution of the target."""
    counts = pd.Series(y).value_counts().sort_index()
    return counts.to_dict(), counts


def select_features(df):
    """Step 11: separate identifiers & target-leakage columns from features and target.
    Note: G3 is used to define the target 'attendance_risk' (Slide 10).
    Including G3 in predictor features X causes Target Leakage (100% artificial accuracy).
    We drop G3 from X so models predict risk realistically from attendance & prior features.
    """
    leakage_cols = ['G3'] if 'G3' in df.columns else []
    X = df.drop(columns=IDENTIFIER_COLS + [TARGET] + leakage_cols, errors='ignore')
    y = df[TARGET]
    return X, y, list(X.columns)


# --------------------------------------------------------------------------
# Visualizations
# --------------------------------------------------------------------------

def make_plots(df):
    """Create the EDA plots required by the spec; return a list of filenames."""
    sns.set_theme(style='darkgrid')
    plots = []

    def save(fig, name):
        path = os.path.join(PLOT_DIR, name)
        fig.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        plots.append(name)

    # 1. Attendance / Absence distribution histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    if 'attendance_percentage' in df.columns:
        ax.hist(df['attendance_percentage'], bins=20, color='#06b6d4', edgecolor='white')
        ax.axvline(72, color='#f59e0b', linestyle='--', label='at-risk threshold (72%)')
        ax.set_title('Attendance percentage distribution')
        ax.set_xlabel('Attendance %')
    elif 'absences' in df.columns:
        ax.hist(df['absences'], bins=20, color='#06b6d4', edgecolor='white')
        ax.axvline(df['absences'].median(), color='#f59e0b', linestyle='--', label=f"median ({df['absences'].median()})")
        ax.set_title('Student Absence Count Distribution (UCI Dataset)')
        ax.set_xlabel('Absences')
    ax.set_ylabel('Students')
    ax.legend()
    save(fig, 'attendance_histogram.png')

    # 2. Category bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    if 'attendance_percentage' in df.columns:
        bins = [0, 60, 70, 80, 90, 100]
        labels = ['<60', '60-70', '70-80', '80-90', '90-100']
        cats = pd.cut(df['attendance_percentage'], bins=bins, labels=labels, right=True)
        cats.value_counts().sort_index().plot(kind='bar', ax=ax, color='#8b5cf6')
        ax.set_title('Students by attendance category')
    elif 'G3' in df.columns:
        bins = [-1, 5, 10, 15, 20]
        labels = ['0-5 (Fail)', '6-10 (Low)', '11-15 (Good)', '16-20 (Exec)']
        cats = pd.cut(df['G3'], bins=bins, labels=labels, right=True)
        cats.value_counts().sort_index().plot(kind='bar', ax=ax, color='#8b5cf6')
        ax.set_title('Students by Grade G3 Category (UCI Dataset)')
    ax.set_ylabel('Students')
    ax.tick_params(axis='x', rotation=0)
    save(fig, 'attendance_categories.png')

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(13, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax, annot_kws={'size': 7})
    ax.set_title('Feature correlation heatmap')
    save(fig, 'correlation_heatmap.png')

    # 4. Risk category distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(['Normal (0)', 'At Risk (1)'], counts.values, color=['#10b981', '#ef4444'])
    ax.set_title('Risk category distribution')
    ax.set_ylabel('Students')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha='center')
    save(fig, 'risk_distribution.png')

    return plots


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

LR_PATH = os.path.join(MODEL_DIR, 'attendance_lr.pkl')

def train_models(X, y):
    """Step: train-test split, Logistic Regression baseline, Random Forest, and XGBoost evaluation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {
        'split': {'train': len(X_train), 'test': len(X_test),
                  'test_size': 0.2, 'random_state': 42},
    }

    # --- Model 1: Logistic Regression Baseline (Slide 9 & 11) -------------
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    results['logistic_regression'] = evaluate(lr, X_test_scaled, y_test)
    with open(LR_PATH, 'wb') as f:
        pickle.dump(lr, f)

    # --- Model 2: Random Forest (Slide 9 & 11) ---------------------------
    rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                class_weight='balanced')
    rf.fit(X_train, y_train)
    results['random_forest'] = evaluate(rf, X_test, y_test)
    with open(RF_PATH, 'wb') as f:
        pickle.dump(rf, f)

    # --- Model 3: XGBoost -------------------------------------------------
    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                        random_state=42, eval_metric='logloss',
                        scale_pos_weight=neg / max(pos, 1))
    xgb.fit(X_train, y_train)
    results['xgboost'] = evaluate(xgb, X_test, y_test)
    with open(XGB_PATH, 'wb') as f:
        pickle.dump(xgb, f)

    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump({'feature_columns': list(X.columns),
                     'class_distribution': {str(k): int(v) for k, v in
                                            results['class_distribution'].items()}
                     if 'class_distribution' in results else None}, f)

    # Feature-importance plots.
    make_importance_plot(lr, X.columns, 'lr_feature_importance.png')
    make_importance_plot(rf, X.columns, 'rf_feature_importance.png')
    make_importance_plot(xgb, X.columns, 'xgb_feature_importance.png')

    return results


def evaluate(model, X_test, y_test):
    """Accuracy, precision, recall, F1, confusion matrix, feature importance."""
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True,
                                   zero_division=0)

    # Weighted metrics for the comparison table.
    weighted = {
        'accuracy': round(float(accuracy_score(y_test, pred)), 4),
        'precision': round(float(precision_score(y_test, pred,
                                                 average='weighted', zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, pred,
                                           average='weighted', zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, pred, average='weighted',
                                   zero_division=0)), 4),
    }
    # Recall for the at-risk class specifically: missing an at-risk student is
    # more costly than flagging a normal student, so this is what we watch.
    at_risk = report.get('1', {})
    weighted['recall_at_risk'] = round(float(at_risk.get('recall', 0.0)), 4)
    weighted['precision_at_risk'] = round(float(at_risk.get('precision', 0.0)), 4)
    weighted['confusion_matrix'] = cm.tolist()
    weighted['classification_report'] = report
    if hasattr(model, 'feature_importances_'):
        importances = [float(v) for v in model.feature_importances_]
    elif hasattr(model, 'coef_'):
        importances = [float(v) for v in np.abs(model.coef_[0])]
    else:
        importances = []

    feat_names = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else []
    weighted['feature_importance'] = dict(zip(feat_names, importances))
    return weighted


def make_importance_plot(model, feature_names, filename):
    if hasattr(model, 'feature_importances_'):
        vals = model.feature_importances_
    elif hasattr(model, 'coef_'):
        vals = np.abs(model.coef_[0])
    else:
        return
    imp = pd.Series(vals, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    imp.tail(15).plot(kind='barh', ax=ax, color='#06b6d4')
    ax.set_title('Feature Importance / Coefficients')
    ax.set_xlabel('Importance / |Coefficient|')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=100, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_metrics_to_db(results):
    """Store the evaluation results for the UI comparison table."""
    conn = get_db()
    cur = conn.cursor()
    for model_name, key in (('Random Forest', 'random_forest'),
                            ('XGBoost', 'xgboost')):
        m = results[key]
        cur.execute("""
            INSERT OR REPLACE INTO model_metrics
                (model_name, accuracy, precision_score, recall_score,
                 recall_at_risk, f1_score, confusion_matrix,
                 classification_report, feature_importance, trained_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (model_name,
              m['accuracy'], m['precision'], m['recall'],
              m['recall_at_risk'], m['f1'],
              json.dumps(m['confusion_matrix']),
              json.dumps(m.get('classification_report', {})),
              json.dumps(m.get('feature_importance', {}))))
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def run_pipeline():
    """Execute the full pipeline; returns a dict the Flask app can serialize."""
    print("=" * 70)
    print("SMART ATTENDANCE AI - ML PIPELINE (attendance-risk prediction)")
    print("=" * 70)

    summary = {}

    df = load_dataset()
    print(f"\n[1] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns "
          f"from {os.path.basename(CSV_PATH)}")
    print("\n[2] First 5 rows:\n", df.head().to_string())
    print("\n[3] Column names:", list(df.columns))
    print("\n[4] Data types & missing values:\n", check_shape_and_types(df))

    df, clean_notes = clean(df)
    print("\n[5] Cleaning:\n" + clean_notes)

    df = engineer_features(df)
    print(f"\n[6] Feature engineering added {ENGINEERED_FEATURES}")

    # EDA plots are drawn BEFORE one-hot encoding so the categorical columns
    # (department, year, section) still exist for the group-by charts.
    plots = make_plots(df)
    print(f"\n[7] Saved {len(plots)} visualizations to static/plots/")

    df = encode_categoricals(df)
    print(f"\n[8] One-hot encoded {CATEGORICAL_FEATURES} -> "
          f"{len(df.columns) - len(IDENTIFIER_COLS) - 1} feature columns")

    X, y, feature_cols = select_features(df)
    dist, _ = check_class_distribution(y)
    print("\n[9] Class distribution:", dist)
    summary['class_distribution'] = {str(k): int(v) for k, v in dist.items()}

    results = train_models(X, y)
    results['class_distribution'] = summary['class_distribution']
    summary['plots'] = plots
    summary['feature_columns'] = feature_cols

    print(f"\n[10] Train-test split: {results['split']['train']} train / "
          f"{results['split']['test']} test (stratified, random_state=42)")
    for name in ('random_forest', 'xgboost'):
        m = results[name]
        print(f"\n--- {name.upper()} ---")
        print(f"    Accuracy: {m['accuracy']:.4f} | Precision: {m['precision']:.4f} "
              f"| Recall: {m['recall']:.4f} | F1: {m['f1']:.4f}")
        print(f"    Recall (At Risk class): {m['recall_at_risk']:.4f}")
        print(f"    Confusion matrix [[TN FP] [FN TP]]: {m['confusion_matrix']}")

    save_metrics_to_db(results)
    print("\n[11] Metrics stored in SQLite (model_metrics).")

    # Add model fitting concepts analysis (Underfitting vs Overfitting vs Good Fitting)
    fitting_res = evaluate_fitting_concepts(X, y)
    summary['fitting_concepts'] = fitting_res
    print("\n[12] Model Fitting Concepts evaluated (Underfitted, Overfitted, Good Fitted models).")

    summary.update(results)
    return summary


def evaluate_fitting_concepts(X, y):
    """Train and compare 3 models demonstrating Underfitting, Overfitting, and Good Fitting concepts.
    
    1. Underfitted Model: DecisionTree max_depth=1 (High Bias - fails to capture patterns)
    2. Overfitted Model: DecisionTree max_depth=25, min_samples_split=2 (High Variance - memorizes noise)
    3. Good Fitted Model: Tuned RandomForest max_depth=8 (Optimal Balance - generalizes well)
    """
    from sklearn.tree import DecisionTreeClassifier
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models_config = {
        'underfitting': {
            'name': 'Underfitted Model (Decision Tree max_depth=1)',
            'model': DecisionTreeClassifier(max_depth=1, random_state=42),
            'concept': 'Underfitting (High Bias)',
            'description': 'Model is too simple to capture patterns. Low accuracy on both Train and Test data.'
        },
        'overfitting': {
            'name': 'Overfitted Model (Decision Tree max_depth=25)',
            'model': DecisionTreeClassifier(max_depth=25, min_samples_split=2, random_state=42),
            'concept': 'Overfitting (High Variance)',
            'description': 'Model memorizes training data including noise. Perfect on Train data (~100%), but drops on Test data.'
        },
        'good_fitting': {
            'name': 'Good Fitted Model (Random Forest max_depth=8)',
            'model': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
            'concept': 'Good Fitting (Balanced)',
            'description': 'Optimal bias-variance balance. Achieves high performance on both Train and Test data.'
        }
    }

    results = {}
    chart_data = []

    for key, item in models_config.items():
        clf = item['model']
        clf.fit(X_train, y_train)

        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        train_f1 = f1_score(y_train, train_pred, zero_division=0)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)

        train_rec = recall_score(y_train, train_pred, zero_division=0)
        test_rec = recall_score(y_test, test_pred, zero_division=0)

        gap = abs(train_acc - test_acc)

        res = {
            'key': key,
            'name': item['name'],
            'concept': item['concept'],
            'description': item['description'],
            'train_accuracy': round(float(train_acc), 4),
            'test_accuracy': round(float(test_acc), 4),
            'train_f1': round(float(train_f1), 4),
            'test_f1': round(float(test_f1), 4),
            'train_recall': round(float(train_rec), 4),
            'test_recall': round(float(test_rec), 4),
            'generalization_gap': round(float(gap), 4)
        }
        results[key] = res
        chart_data.append(res)

    # Plot comparison chart for fitting concepts
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = [d['concept'].split(' ')[0] for d in chart_data]
    x = np.arange(len(categories))
    width = 0.35

    train_accs = [d['train_accuracy'] for d in chart_data]
    test_accs = [d['test_accuracy'] for d in chart_data]

    rects1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', color='#6366f1')
    rects2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='#06b6d4')

    ax.set_ylabel('Accuracy Score')
    ax.set_title('Model Fitting Comparison: Underfitting vs Overfitting vs Good Fitting')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add text labels on top of bars
    for bar in rects1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in rects2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fitting_plot_path = os.path.join(PLOT_DIR, 'fitting_comparison.png')
    fig.savefig(fitting_plot_path, dpi=120)
    plt.close(fig)

    return results


# --------------------------------------------------------------------------
# Prediction helpers (used by the Flask risk-prediction endpoints)
# --------------------------------------------------------------------------

def _load_model_and_features():
    if not (os.path.exists(RF_PATH) and os.path.exists(FEATURES_PATH)):
        return None, None
    with open(RF_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        meta = pickle.load(f)
    return model, meta['feature_columns']


def predict_from_row(row_dict):
    """Predict risk for one student given a full feature dict.

    The row is aligned to the training feature columns; missing columns are
    filled with the training medians (a documented imputation choice).
    """
    model, feature_cols = _load_model_and_features()
    if model is None:
        return {'error': 'Models not trained yet. Run the ML pipeline first.'}

    df = engineer_features(pd.read_csv(CSV_PATH))
    medians = df[NUMERIC_FEATURES + ENGINEERED_FEATURES].median()

    row = {f: np.nan for f in feature_cols}
    for k, v in row_dict.items():
        if k in row:
            row[k] = v

    # Derive engineered features when the raw inputs are available.
    if (not pd.isna(row.get('attendance_percentage'))
            and not pd.isna(row.get('previous_semester_percentage'))):
        row['attendance_gap'] = row['attendance_percentage'] - row['previous_semester_percentage']
    if (not pd.isna(row.get('assignment_score'))
            and not pd.isna(row.get('internal_marks'))):
        row['avg_academic_score'] = (row['assignment_score'] + row['internal_marks']) / 2

    # Missing numeric features: training medians (documented imputation).
    # Missing one-hot columns: 0 (no category information for the student).
    for col in feature_cols:
        if pd.isna(row[col]):
            row[col] = medians[col] if col in medians.index else 0

    proba = model.predict_proba(pd.DataFrame([row])[feature_cols])[0]
    pred = int(np.argmax(proba))
    return {
        'predicted_risk': pred,
        'label': 'At Risk' if pred == 1 else 'Normal',
        'probability_at_risk': round(float(proba[1]), 4),
    }


def top_at_risk(n=10):
    """Rank students from the demo dataset by predicted at-risk probability."""
    model, feature_cols = _load_model_and_features()
    if model is None:
        return {'error': 'Models not trained yet. Run the ML pipeline first.'}
    df = load_dataset()
    # Same column order as the training pipeline: engineer, then encode.
    X, y, cols = select_features(encode_categoricals(engineer_features(df.copy())))
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        'roll_number': df['roll_number'],
        'attendance_percentage': df['attendance_percentage'],
        'consecutive_absences': df['consecutive_absences'],
        'actual': y.values,
        'predicted_probability': np.round(proba, 3),
    })
    out = out.sort_values('predicted_probability', ascending=False).head(n)
    return out.to_dict('records')


if __name__ == '__main__':
    run_pipeline()
