"""Train the face-recognition benchmark models (Eigenfaces + SVM / KNN).

This trains identity classifiers on the public LFW (Labeled Faces in the Wild)
benchmark dataset. It is a demonstration of the face-recognition approach -
the models recognise LFW identities, NOT students registered in this system.
Recognising actual students requires training on the collected face samples
(dataset/<roll_number>/) and is listed as post-mid-review work.

This module is independent of the attendance-risk ML pipeline (ml_pipeline.py),
which predicts attendance risk from attendance/academic features and never
touches face images.

Note: the models are only trained once. fetch_lfw_people downloads the dataset
from the internet on first use, so this script needs a network connection.

Run:  python model_trainer.py
"""

import os
import cv2
import numpy as np
import pickle

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from database import get_db

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_eval_models():
    print("Loading LFW benchmark dataset (downloads on first run)...")
    lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
    X, y = lfw.data, lfw.target
    target_names = lfw.target_names
    h, w = lfw.images.shape[1], lfw.images.shape[2]
    print(f"LFW loaded: {X.shape[0]} images, {len(target_names)} identities, "
          f"image shape ({h}, {w})")

    # Same preprocessing used at inference time (face_recognizer.py):
    # normalize to 0-255, equalize histogram for lighting invariance, resize
    # to a fixed 64x64. Keeping the two in sync matters, otherwise PCA/SVM/KNN
    # pickles are incompatible with the live recognizer.
    X_processed = preprocess_faces(X.reshape(-1, h, w), target_dim=(64, 64))

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.25, random_state=42, stratify=y)

    n_components = 150
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    with open(os.path.join(MODEL_DIR, 'pca_model.pkl'), 'wb') as f:
        pickle.dump(pca, f)

    # --- Model 1: Eigenfaces (PCA) + SVM ----------------------------------
    print("\n--- Training Eigenfaces (PCA) + SVM ---")
    svm = SVC(kernel='rbf', C=1000.0, gamma=0.005, probability=True)
    svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(X_test_pca)
    store_metrics('Eigenfaces + SVM', y_test, y_pred)
    with open(os.path.join(MODEL_DIR, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(svm, f)

    # --- Model 2: Eigenfaces (PCA) + KNN ----------------------------------
    print("\n--- Training Eigenfaces (PCA) + KNN ---")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)
    store_metrics('Eigenfaces + KNN', y_test, y_pred)
    with open(os.path.join(MODEL_DIR, 'knn_model.pkl'), 'wb') as f:
        pickle.dump(knn, f)

    with open(os.path.join(MODEL_DIR, 'target_names.pkl'), 'wb') as f:
        pickle.dump(target_names, f)

    print("\nFace recognition benchmark models saved to models/.")


def preprocess_faces(images, target_dim=(64, 64)):
    """Normalize, equalize, resize - must match face_recognizer preprocessing."""
    processed = []
    for img in images:
        norm = cv2.normalize(img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        eq = cv2.equalizeHist(norm)
        resized = cv2.resize(eq, target_dim)
        processed.append(resized.flatten())
    return np.array(processed)


def store_metrics(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    n_classes = len(set(y_true))

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO face_model_metrics
            (model_name, accuracy, precision_score, recall_score, f1_score,
             n_classes, trained_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (model_name, acc, prec, rec, f1, n_classes))
    conn.commit()
    conn.close()

    print(f"    {model_name}: accuracy {acc*100:.2f}% | precision {prec:.3f} "
          f"| recall {rec:.3f} | f1 {f1:.3f} | {n_classes} classes")


if __name__ == '__main__':
    train_and_eval_models()
