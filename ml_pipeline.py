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

import numpy as np
if not hasattr(np, 'long'):
    np.long = int
if not hasattr(np, 'ulong'):
    np.ulong = int
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

def load_dataset():
    """Step 1-3: load CSV, show shape and column names."""
    df = pd.read_csv(CSV_PATH)
    # pandas infers roll numbers as int; they are identifiers, not numbers.
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

    # 6. Missing values: this synthetic dataset should have none, but handle
    #    them anyway so the pipeline works with real data later.
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        notes.append(f"Filled {df.isnull().sum().sum()} missing values "
                     f"in {missing_cols} with column medians.")
        for col in missing_cols:
            df[col] = df[col].fillna(df[col].median())
    else:
        notes.append("No missing values found - nothing to fill.")

    # 7. Duplicate records.
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        notes.append(f"Removed {n_dup} duplicate rows.")
    else:
        notes.append("No duplicate rows found.")

    # 8. Invalid values: percentages must be in [0, 100]; counts non-negative.
    for col in ['attendance_percentage', 'absence_percentage',
                'attendance_last_7_days', 'attendance_last_30_days',
                'previous_semester_percentage']:
        bad = df[(df[col] < 0) | (df[col] > 100)]
        if len(bad):
            notes.append(f"Clamped {len(bad)} out-of-range values in {col}.")
            df.loc[df[col] < 0, col] = 0
            df.loc[df[col] > 100, col] = 100
    for col in ['total_classes', 'classes_attended', 'classes_absent',
                'consecutive_absences', 'number_of_late_arrivals']:
        bad = df[df[col] < 0]
        if len(bad):
            notes.append(f"Removed {len(bad)} rows with negative {col}.")
            df = df[df[col] >= 0].reset_index(drop=True)

    notes.append(f"Rows before cleaning: {n_before}, after: {len(df)}")
    return df, "\n".join(notes)


def engineer_features(df):
    """Step: feature engineering.

    attendance_gap        = attendance_percentage - previous_semester_percentage
                          Captures whether the student's attendance is trending
                          up or down versus last semester. A student who is
                          dropping sharply is more likely to become at risk.

    avg_academic_score    = mean(assignment_score, internal_marks)
                          A single summary of current academic standing, so the
                          model can weigh it together with attendance.
    """
    df['attendance_gap'] = df['attendance_percentage'] - df['previous_semester_percentage']
    df['avg_academic_score'] = df[['assignment_score', 'internal_marks']].mean(axis=1)
    return df


def encode_categoricals(df):
    """Step 9-10: one-hot encode categorical variables for the models."""
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dtype=int)
    return df


def check_class_distribution(y):
    """Step 12: class distribution of the target."""
    counts = pd.Series(y).value_counts().sort_index()
    return counts.to_dict(), counts


def select_features(df):
    """Step 11: separate identifiers from features and target.

    Returns (X, y, feature_columns)."""
    X = df.drop(columns=IDENTIFIER_COLS + [TARGET], errors='ignore')
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

    # 1. Attendance distribution histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['attendance_percentage'], bins=20, color='#06b6d4', edgecolor='white')
    ax.axvline(72, color='#f59e0b', linestyle='--', label='at-risk threshold (72%)')
    ax.set_title('Attendance percentage distribution')
    ax.set_xlabel('Attendance %')
    ax.set_ylabel('Students')
    ax.legend()
    save(fig, 'attendance_histogram.png')

    # 2. Attendance category bar chart (buckets)
    bins = [0, 60, 70, 80, 90, 100]
    labels = ['<60', '60-70', '70-80', '80-90', '90-100']
    cats = pd.cut(df['attendance_percentage'], bins=bins, labels=labels, right=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    cats.value_counts().sort_index().plot(kind='bar', ax=ax, color='#8b5cf6')
    ax.set_title('Students by attendance category')
    ax.set_xlabel('Attendance %')
    ax.set_ylabel('Students')
    ax.tick_params(axis='x', rotation=0)
    save(fig, 'attendance_categories.png')

    # 3. Present vs absent counts (aggregated over the dataset)
    present = int(df['classes_attended'].sum())
    absent = int(df['classes_absent'].sum())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Present', 'Absent'], [present, absent], color=['#10b981', '#ef4444'])
    ax.set_title('Total present vs absent class-days')
    ax.set_ylabel('Class-days')
    for i, v in enumerate([present, absent]):
        ax.text(i, v + 200, f'{v:,}', ha='center')
    save(fig, 'present_vs_absent.png')

    # 4. Department-wise mean attendance
    fig, ax = plt.subplots(figsize=(8, 4))
    df.groupby('department')['attendance_percentage'].mean().sort_values().plot(
        kind='barh', ax=ax, color='#06b6d4')
    ax.set_title('Mean attendance by department')
    ax.set_xlabel('Mean attendance %')
    save(fig, 'department_attendance.png')

    # 5. Year-wise mean attendance
    fig, ax = plt.subplots(figsize=(8, 4))
    df.groupby('year')['attendance_percentage'].mean().sort_index().plot(
        kind='bar', ax=ax, color='#3b82f6')
    ax.set_title('Mean attendance by year')
    ax.set_ylabel('Mean attendance %')
    ax.tick_params(axis='x', rotation=0)
    save(fig, 'year_attendance.png')

    # 6. Section-wise mean attendance
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby('section')['attendance_percentage'].mean().sort_index().plot(
        kind='bar', ax=ax, color='#f59e0b')
    ax.set_title('Mean attendance by section')
    ax.set_ylabel('Mean attendance %')
    ax.tick_params(axis='x', rotation=0)
    save(fig, 'section_attendance.png')

    # 7. Attendance vs academic performance (scatter)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df['attendance_percentage'], df['internal_marks'],
               c=df[TARGET], cmap='coolwarm', alpha=0.6, s=18)
    ax.set_title('Attendance vs internal marks (colored by risk)')
    ax.set_xlabel('Attendance %')
    ax.set_ylabel('Internal marks')
    save(fig, 'attendance_vs_marks.png')

    # 8. Correlation heatmap (the most important mid-review visualization)
    fig, ax = plt.subplots(figsize=(13, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, ax=ax, annot_kws={'size': 7})
    ax.set_title('Feature correlation heatmap')
    save(fig, 'correlation_heatmap.png')

    # 10. Risk category distribution
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

def train_models(X, y):
    """Step: train-test split, Random Forest and XGBoost, full evaluation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {
        'split': {'train': len(X_train), 'test': len(X_test),
                  'test_size': 0.2, 'random_state': 42},
    }

    # --- Model 1: Random Forest ------------------------------------------
    # An ensemble of many decision trees; predictions are combined by majority
    # vote. Strong on tabular data, handles nonlinearity, no scaling needed,
    # and exposes feature importances. class_weight='balanced' compensates for
    # the imbalanced classes.
    rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                class_weight='balanced')
    rf.fit(X_train, y_train)
    results['random_forest'] = evaluate(rf, X_test, y_test)
    with open(RF_PATH, 'wb') as f:
        pickle.dump(rf, f)

    # --- Model 2: XGBoost -------------------------------------------------
    # Gradient boosting: builds trees sequentially, each new tree focuses on
    # the errors of the previous ones. scale_pos_weight compensates for the
    # minority (at-risk) class.
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
    weighted['feature_importance'] = dict(
        zip(model.feature_names_in_.tolist()
            if hasattr(model, 'feature_names_in_') else [],
            [float(v) for v in model.feature_importances_]))
    return weighted


def make_importance_plot(model, feature_names, filename):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    imp.tail(15).plot(kind='barh', ax=ax, color='#06b6d4')
    ax.set_title('Feature importance')
    ax.set_xlabel('Importance')
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
