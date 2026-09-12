"""Face module: Haar-cascade face detection, face-sample collection, and
benchmark face recognition.

Two separate responsibilities in this project (per the spec):

  1. Face detection + face-sample collection  -> used by Module 2
     (capture webcam frames and save them under dataset/<roll_number>/)

  2. Face recognition -> "who is this person?" The trained recognition models
     (PCA + SVM / PCA + KNN in models/) are trained on the public LFW
     benchmark dataset, NOT on the students of this system. They are used here
     as a demonstration of the face-recognition approach. Recognising the
     actual registered students requires training on the collected samples and
     is listed as post-mid-review work in the README.

The attendance-risk ML models (Random Forest / XGBoost in ml_pipeline.py) are
a completely separate module and are never trained on face images.
"""

import os
import base64
import pickle
import datetime

import cv2
import numpy as np

from database import get_db

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

CASCADE_PATH = os.path.join(BASE_DIR, 'cascades', 'haarcascade_frontalface_default.xml')

def get_face_cascade():
    if hasattr(cv2, 'CascadeClassifier') and os.path.exists(CASCADE_PATH):
        try:
            cascade = cv2.CascadeClassifier(CASCADE_PATH)
            if not cascade.empty():
                return cascade
        except Exception as e:
            print(f"Cascade load note: {e}")
    return None

face_cascade = get_face_cascade()


# --------------------------------------------------------------------------
# Face detection + sample collection (Module 2 - webcam capture)
# --------------------------------------------------------------------------

def collect_face_sample(image_bytes, roll_number):
    """Detect a face in a webcam frame and save it to dataset/<roll_number>/.

    Returns a dict; on success includes the saved path and the student's
    current sample count. Rejects frames with no detectable face.
    """
    frame = decode_image(image_bytes)
    if frame is None:
        return {'error': 'Invalid image payload'}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2,
                                          minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return {'error': 'No face detected in the frame. Try again with better lighting.'}

    student_dir = os.path.join(DATASET_DIR, str(roll_number))
    os.makedirs(student_dir, exist_ok=True)

    # Keep the largest face in the frame.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y + h, x:x + w]
    face_roi = cv2.resize(face_roi, (160, 160))

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'{timestamp}.jpg'
    path = os.path.join(student_dir, filename)
    cv2.imwrite(path, face_roi)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO face_samples (roll_number, image_path) VALUES (?, ?)",
                (str(roll_number), path))
    conn.commit()
    count = cur.execute("SELECT COUNT(*) FROM face_samples WHERE roll_number = ?",
                        (str(roll_number),)).fetchone()[0]
    conn.close()

    return {
        'status': 'success',
        'saved_to': path,
        'sample_count': count,
    }


def get_sample_counts():
    """Number of collected face samples per registered student."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT roll_number, COUNT(*) AS count
        FROM face_samples GROUP BY roll_number ORDER BY roll_number
    """)
    counts = {r['roll_number']: r['count'] for r in cur.fetchall()}
    conn.close()
    return counts


# --------------------------------------------------------------------------
# Benchmark face recognition (LFW-trained models - demonstration only)
# --------------------------------------------------------------------------

def load_recognition_models():
    pca_path = os.path.join(MODEL_DIR, 'pca_model.pkl')
    svm_path = os.path.join(MODEL_DIR, 'svm_model.pkl')
    knn_path = os.path.join(MODEL_DIR, 'knn_model.pkl')
    names_path = os.path.join(MODEL_DIR, 'target_names.pkl')
    needed = [pca_path, svm_path, knn_path, names_path]
    if not all(os.path.exists(p) for p in needed):
        return None, None, None, None
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    with open(svm_path, 'rb') as f:
        svm = pickle.load(f)
    with open(knn_path, 'rb') as f:
        knn = pickle.load(f)
    with open(names_path, 'rb') as f:
        target_names = pickle.load(f)
    return pca, svm, knn, target_names


def preprocess_single_face(face_gray, target_dim=(64, 64)):
    """Same preprocessing used at training time (normalize, equalize, resize)."""
    norm = cv2.normalize(face_gray, None, alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    eq = cv2.equalizeHist(norm)
    resized = cv2.resize(eq, target_dim)
    return resized.flatten().reshape(1, -1)


def decode_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def detect_and_recognize_frame(image_bytes, model_choice='svm'):
    """Detect faces and recognize them with the LFW-trained benchmark models.

    Always returns a dict. If the models have not been trained, returns an
    error message in the dict (never a bare tuple).
    """
    frame = decode_image(image_bytes)
    if frame is None:
        return {'error': 'Invalid image payload'}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2,
                                          minNeighbors=5, minSize=(30, 30))

    pca, svm, knn, target_names = load_recognition_models()
    if pca is None:
        return {'error': 'Recognition models not found. Run model_trainer.py first.'}

    model = svm if model_choice == 'svm' else knn
    model_label = ('Eigenfaces + SVM' if model_choice == 'svm'
                   else 'Eigenfaces + KNN')

    results = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        vec = preprocess_single_face(face_roi)
        feat = pca.transform(vec)
        probs = model.predict_proba(feat)[0]
        best = int(np.argmax(probs))
        conf = float(probs[best])
        name = target_names[best]

        results.append({
            'bbox': [int(x), int(y), int(w), int(h)],
            'predicted_identity': name,   # an LFW identity, not a registered student
            'confidence': round(conf, 4),
        })
        cv2.rectangle(frame, (x, y), (x + w, y + h), (34, 197, 94), 2)
        cv2.putText(frame, f'{name} ({conf*100:.1f}%)', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        'faces_detected': len(faces),
        'results': results,
        'annotated_image': f'data:image/jpeg;base64,{annotated_b64}',
        'model_used': model_label,
        'note': ('Benchmark demo: identities come from the LFW training set, '
                 'not from registered students.'),
    }
