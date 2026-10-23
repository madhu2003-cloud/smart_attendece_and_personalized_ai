"""Face module: Haar-cascade adaptive face detection, face-sample collection, and
real-time student face recognition with registered sample matching and automatic attendance marking.

Features:
  1. Multi-tier Adaptive Face Detection (CLAHE + tiered Haar cascades for robust webcam capture)
  2. Illumination-Invariant & Translation-Invariant Multi-Feature Face Descriptor:
     - CLAHE lighting equalization
     - Multi-region translation-invariant template correlation (eye & center regions)
     - Multi-scale Spatial Local Binary Patterns (LBP) texture histograms
     - Spatial contrast & gradient magnitude feature grid
  3. Automatic Attendance Logging when face matches registered samples (status 'Present', source 'face_ai').
  4. Non-matching faces or frames with no face detected are clearly reported with specific diagnostics.
"""

import os
import glob
import base64
import datetime
import cv2
import numpy as np

from database import get_db, format_roll_number

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
CASCADE_PATH = os.path.join(BASE_DIR, 'cascades', 'haarcascade_frontalface_default.xml')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Empirically calibrated similarity threshold (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.58


def get_face_cascade():
    if hasattr(cv2, 'CascadeClassifier') and os.path.exists(CASCADE_PATH):
        try:
            cascade = cv2.CascadeClassifier(CASCADE_PATH)
            if not cascade.empty():
                return cascade
        except Exception as e:
            print(f"Cascade load note: {e}")
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


face_cascade = get_face_cascade()


def decode_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------
# Multi-Tier Adaptive Face Detection
# --------------------------------------------------------------------------

def detect_faces_adaptive(gray):
    """Detect faces across multiple sensitivity tiers to handle varying
    webcam resolutions, lighting conditions, and slight face tilts.
    """
    # Tier 1: Standard detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(45, 45))
    if len(faces) > 0:
        return faces

    # Tier 2: Relaxed sensitivity for webcam streams
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(35, 35))
    if len(faces) > 0:
        return faces

    # Tier 3: CLAHE lighting-equalized detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    faces = face_cascade.detectMultiScale(eq, scaleFactor=1.08, minNeighbors=3, minSize=(35, 35))
    return faces if len(faces) > 0 else []


# --------------------------------------------------------------------------
# Facial Feature Extraction & Matching
# --------------------------------------------------------------------------

def extract_face_descriptor(face_roi):
    """Extract an illumination-invariant, multi-scale facial representation.
    Combines:
      - CLAHE lighting equalization
      - 8x8 cell grid local contrast & gradient magnitude features
      - Local Binary Pattern (LBP) texture histograms (4x4 spatial blocks)
      - Standardized 128x128 enhanced grayscale matrix
    """
    if len(face_roi.shape) == 3:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_roi.copy()

    # Standardize to 128x128
    gray = cv2.resize(gray, (128, 128))

    # 1. CLAHE for robust illumination invariance
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Gradient magnitude & direction (Sobel)
    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    mag, _ = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # 3. 8x8 Grid spatial contrast & edge density features (mean-centered)
    gh, gw = 16, 16
    grid_feats = []
    for r in range(8):
        for c in range(8):
            cell = enhanced[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
            m_cell = mag[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
            grid_feats.append(float(np.mean(cell)))
            grid_feats.append(float(np.std(cell)))
            grid_feats.append(float(np.mean(m_cell)))
    g_vec = np.array(grid_feats, dtype=np.float32)
    g_vec = g_vec - np.mean(g_vec)
    g_norm = np.linalg.norm(g_vec)
    if g_norm > 0:
        g_vec = g_vec / g_norm

    # 4. Local Binary Pattern (LBP) texture calculation
    h, w = enhanced.shape
    pad = np.pad(enhanced, 1, mode='edge')
    center = pad[1:-1, 1:-1]
    lbp = np.zeros_like(enhanced, dtype=np.uint8)
    offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    for w_idx, (dy, dx) in enumerate(offsets):
        neighbor = pad[1+dy:h+1+dy, 1+dx:w+1+dx]
        lbp += ((neighbor >= center) * (1 << w_idx)).astype(np.uint8)

    # 4x4 spatial blocks of normalized LBP histograms
    lbp_hists = []
    bh, bw = 32, 32
    for br in range(4):
        for bc in range(4):
            b_cell = lbp[br*bh:(br+1)*bh, bc*bw:(bc+1)*bw]
            hist, _ = np.histogram(b_cell, bins=16, range=(0, 256))
            h_sum = hist.sum()
            hist_norm = (hist / h_sum).astype(np.float32) if h_sum > 0 else np.zeros(16, dtype=np.float32)
            lbp_hists.append(hist_norm)

    return {
        'enhanced': enhanced,
        'grid': g_vec,
        'lbp': lbp_hists,
    }


def compare_descriptors(descA, descB):
    """Compute normalized composite similarity score (0.0 to 1.0) between two face descriptors.
    Uses translation-invariant multi-region template correlation to handle minor face movements.
    """
    # 1. Spatial grid contrast & edge correlation
    sim_grid = max(0.0, float(np.dot(descA['grid'], descB['grid'])))

    # 2. LBP spatial block histogram intersection
    histsA = descA['lbp']
    histsB = descB['lbp']
    sim_lbp = float(np.mean([np.sum(np.minimum(ha, hb)) for ha, hb in zip(histsA, histsB)]))

    # 3. Translation-invariant multi-region template correlation (handles slight head shifts & tilts)
    # Sub-region 1: Upper face (eyes / eyebrow features)
    eye_sub = descA['enhanced'][20:65, 20:108]
    res_eye = cv2.matchTemplate(descB['enhanced'][12:75, 12:116], eye_sub, cv2.TM_CCOEFF_NORMED)
    _, score_eye, _, _ = cv2.minMaxLoc(res_eye)

    # Sub-region 2: Central face (eyes + nose structure)
    center_sub = descA['enhanced'][24:104, 24:104]
    res_center = cv2.matchTemplate(descB['enhanced'][14:114, 14:114], center_sub, cv2.TM_CCOEFF_NORMED)
    _, score_center, _, _ = cv2.minMaxLoc(res_center)

    corr_score = 0.5 * max(0.0, float(score_eye)) + 0.5 * max(0.0, float(score_center))

    # Composite weighted similarity
    score = (0.45 * sim_lbp) + (0.35 * corr_score) + (0.20 * sim_grid)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def load_student_sample_descriptors(roll_number):
    """Load all saved face samples for a student and return their descriptors.
    Checks SQLite records, exact directory, and STU-prefixed / stripped variations.
    """
    if not roll_number:
        return []

    roll_formatted = format_roll_number(str(roll_number))
    raw_str = str(roll_number).strip().upper()
    variants = {roll_formatted, raw_str}
    if roll_formatted.startswith('STU'):
        variants.add(roll_formatted[3:])

    image_files = set()

    # 1. From SQLite database table face_samples
    try:
        conn = get_db()
        cur = conn.cursor()
        placeholders = ','.join(['?'] * len(variants))
        cur.execute(f"SELECT image_path FROM face_samples WHERE UPPER(roll_number) IN ({placeholders})", list(variants))
        for r in cur.fetchall():
            p = r['image_path']
            if p and os.path.exists(p):
                image_files.add(p)
        conn.close()
    except Exception as e:
        print(f"DB sample query note: {e}")

    # 2. From dataset directories on disk
    for v in variants:
        s_dir = os.path.join(DATASET_DIR, v)
        if os.path.isdir(s_dir):
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                image_files.update(glob.glob(os.path.join(s_dir, ext)))

    descriptors = []
    for fpath in image_files:
        try:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size > 0:
                descriptors.append(extract_face_descriptor(img))
        except Exception as e:
            print(f"Error loading sample {fpath}: {e}")
    return descriptors


# --------------------------------------------------------------------------
# Face Detection + Sample Collection (Enrollment)
# --------------------------------------------------------------------------

def collect_face_sample(image_bytes, roll_number):
    """Detect a face in a webcam/photo frame and save it to dataset/<roll_number>/."""
    frame = decode_image(image_bytes)
    if frame is None:
        return {'error': 'Invalid image payload'}

    roll_formatted = format_roll_number(str(roll_number))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_adaptive(gray)

    if len(faces) == 0:
        h, w = gray.shape
        x, y, sz = w // 4, h // 4, min(w, h) // 2
        faces = [(x, y, sz, sz)]

    student_dir = os.path.join(DATASET_DIR, roll_formatted)
    os.makedirs(student_dir, exist_ok=True)

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Pad crop slightly for natural facial boundary
    pad_y = int(0.12 * h)
    pad_x = int(0.12 * w)
    y1 = max(0, y - pad_y)
    y2 = min(gray.shape[0], y + h + pad_y)
    x1 = max(0, x - pad_x)
    x2 = min(gray.shape[1], x + w + pad_x)

    face_roi = gray[y1:y2, x1:x2]
    face_roi = cv2.resize(face_roi, (160, 160))

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'{timestamp}.jpg'
    path = os.path.join(student_dir, filename)
    cv2.imwrite(path, face_roi)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO face_samples (roll_number, image_path) VALUES (?, ?)", (roll_formatted, path))
    conn.commit()
    sample_id = cur.lastrowid
    count = cur.execute("SELECT COUNT(*) FROM face_samples WHERE roll_number = ?", (roll_formatted,)).fetchone()[0]
    conn.close()

    # Generate thumbnail base64
    _, buf = cv2.imencode('.jpg', face_roi)
    thumb_b64 = base64.b64encode(buf).decode('utf-8')

    return {
        'status': 'success',
        'sample_id': sample_id,
        'saved_to': path,
        'sample_count': count,
        'total_samples': count,
        'thumbnail': f'data:image/jpeg;base64,{thumb_b64}'
    }


def delete_face_sample(sample_id):
    """Delete a face sample from database and disk."""
    conn = get_db()
    cur = conn.cursor()
    row = cur.execute("SELECT roll_number, image_path FROM face_samples WHERE id = ?", (sample_id,)).fetchone()
    if not row:
        conn.close()
        return {'status': 'error', 'message': 'Sample not found'}

    path = row['image_path']
    roll = row['roll_number']
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"File delete note: {e}")

    cur.execute("DELETE FROM face_samples WHERE id = ?", (sample_id,))
    conn.commit()
    remaining = cur.execute("SELECT COUNT(*) FROM face_samples WHERE roll_number = ?", (roll,)).fetchone()[0]
    conn.close()

    return {
        'status': 'success',
        'message': f'Sample deleted. {remaining} samples remaining for {roll}.',
        'remaining_count': remaining
    }


def get_sample_counts():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT roll_number, COUNT(*) AS count FROM face_samples GROUP BY roll_number ORDER BY roll_number")
    counts = {r['roll_number']: r['count'] for r in cur.fetchall()}
    conn.close()
    return counts


# --------------------------------------------------------------------------
# Live Face Recognition & Sample Matching Attendance
# --------------------------------------------------------------------------

def detect_and_recognize_frame(image_bytes, subject="Machine Learning & AI", student_roll=None):
    """Detect faces in webcam frame, match against registered face sample(s),
    annotate bounding boxes & labels, and log attendance if verified.
    """
    frame = decode_image(image_bytes)
    if frame is None:
        return {'error': 'Invalid image payload'}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_adaptive(gray)

    # If absolutely no face detected in frame, return diagnostic status rather than false mismatch
    if len(faces) == 0:
        _, buffer = cv2.imencode('.jpg', frame)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        return {
            'status': 'no_face',
            'faces_detected': 0,
            'match': False,
            'results': [],
            'annotated_image': f'data:image/jpeg;base64,{annotated_b64}',
            'model_used': 'Adaptive Haar-Cascade + Multi-Feature Descriptor Matcher',
            'message': 'No face detected in camera frame. Please position your face clearly in the camera with adequate lighting.'
        }

    conn = get_db()
    cur = conn.cursor()

    # Determine target student(s) to verify against
    target_student = None
    if student_roll:
        student_roll_formatted = format_roll_number(student_roll)
        row = cur.execute("""
            SELECT roll_number, name FROM students 
            WHERE roll_number = ? OR roll_number = ?
        """, (student_roll_formatted, str(student_roll).strip())).fetchone()
        if not row:
            # Check users table
            row = cur.execute("""
                SELECT username as roll_number, name FROM users 
                WHERE role='student' AND (username = ? OR username = ?)
            """, (student_roll_formatted, str(student_roll).strip())).fetchone()

        if row:
            target_student = dict(row)
        else:
            target_student = {'roll_number': student_roll_formatted, 'name': f"Student {student_roll_formatted}"}

    # Fetch all registered students
    all_students = [dict(s) for s in cur.execute("SELECT roll_number, name FROM students ORDER BY roll_number").fetchall()]
    conn.close()

    results = []
    annotated_frame = frame.copy()
    overall_match = False
    message = ""

    for idx, (x, y, w, h) in enumerate(faces):
        face_roi = gray[y:y + h, x:x + w]
        live_desc = extract_face_descriptor(face_roi)

        # Mode A: Student self-scan with target_student specified
        if target_student:
            roll = target_student['roll_number']
            name = target_student['name']
            samples = load_student_sample_descriptors(roll)

            # Check target student samples
            target_score = 0.0
            if samples:
                scores = [compare_descriptors(live_desc, s) for s in samples]
                target_score = max(scores)

            # Cross-check against all other students to detect cross-account impersonation
            other_best_student = None
            other_best_score = 0.0
            for st in all_students:
                if st['roll_number'] == roll:
                    continue
                o_samples = load_student_sample_descriptors(st['roll_number'])
                if o_samples:
                    o_score = max([compare_descriptors(live_desc, s) for s in o_samples])
                    if o_score > other_best_score:
                        other_best_score = o_score
                        other_best_student = st

            matched = False
            is_impersonation = False

            if len(samples) == 0 and other_best_score < SIMILARITY_THRESHOLD:
                conf = 0.0
                box_color = (0, 165, 255)  # Orange/Amber
                status_txt = "NOT ENROLLED"
                message = f"No face samples registered for {roll} ({name}). Please register your face sample first."
            elif target_score >= SIMILARITY_THRESHOLD and target_score >= other_best_score:
                # Genuine student match!
                conf = round(target_score, 3)
                matched = True
                overall_match = True
                box_color = (34, 197, 94)  # Emerald Green
                status_txt = f"{roll} {name} ({conf*100:.1f}%)"
                message = f"Face Matched! Identity verified as {roll} - {name} ({conf*100:.1f}%). Attendance marked Present."
            elif other_best_student and other_best_score >= SIMILARITY_THRESHOLD and other_best_score > target_score:
                # Cross-account mismatch / impersonation attempt
                is_impersonation = True
                conf = round(other_best_score, 3)
                box_color = (59, 59, 239)  # Crimson Rose
                detected_name = other_best_student['name']
                detected_roll = other_best_student['roll_number']
                status_txt = f"MISMATCH: {detected_roll}"
                message = f"IDENTITY MISMATCH: Biometric scan detected you are {detected_name} ({detected_roll}, {conf*100:.1f}%), but you are in {name}'s account ({roll}). You cannot mark attendance for another student's account!"
            else:
                conf = round(target_score, 3)
                box_color = (59, 59, 239)  # Crimson Rose
                status_txt = f"MISMATCH ({conf*100:.1f}%)"
                message = f"Biometric Verification Failed ({conf*100:.1f}%). Face does not match registered samples for {name} ({roll}). Attendance NOT marked."

            # Mark attendance ONLY if matched and NOT impersonating
            marked = False
            if matched and not is_impersonation:
                try:
                    conn = get_db()
                    cur = conn.cursor()
                    today_str = datetime.date.today().isoformat()
                    time_str = datetime.datetime.now().strftime('%H:%M:%S')
                    cur.execute("""
                        INSERT INTO attendance (roll_number, subject, date, time, status, source)
                        VALUES (?, ?, ?, ?, 'Present', 'face_ai')
                        ON CONFLICT(roll_number, subject, date) DO UPDATE SET status='Present', time=excluded.time, source='face_ai'
                    """, (roll, subject, today_str, time_str))
                    conn.commit()
                    conn.close()
                    marked = True
                except Exception as e:
                    print(f"Attendance mark note: {e}")

            # Approximated biometric landmarks for HUD reticle
            landmarks = {
                'left_eye': [int(x + w * 0.32), int(y + h * 0.38)],
                'right_eye': [int(x + w * 0.68), int(y + h * 0.38)],
                'nose': [int(x + w * 0.50), int(y + h * 0.58)],
                'mouth': [int(x + w * 0.50), int(y + h * 0.78)]
            }

            results.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'landmarks': landmarks,
                'roll_number': roll,
                'student_name': name,
                'predicted_identity': f"{roll} - {name}" if matched else (f"Detected: {other_best_student['roll_number']} - {other_best_student['name']}" if is_impersonation else "Unrecognized"),
                'confidence': conf,
                'confidence_pct': round(conf * 100, 1),
                'match': matched,
                'is_impersonation': is_impersonation,
                'detected_other': other_best_student if is_impersonation else None,
                'samples_count': len(samples),
                'attendance_marked': marked
            })

            # Draw bounding box & badge
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 3)
            cv2.rectangle(annotated_frame, (x, y - 32), (x + w, y), box_color, -1)
            cv2.putText(annotated_frame, status_txt, (x + 6, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        # Mode B: Kiosk / Multi-student scan across all enrolled students
        else:
            best_match = None
            highest_score = 0.0

            for st in all_students:
                r_num = st['roll_number']
                samples = load_student_sample_descriptors(r_num)
                if samples:
                    score = max([compare_descriptors(live_desc, s) for s in samples])
                    if score > highest_score:
                        highest_score = score
                        best_match = st

            if best_match and highest_score >= SIMILARITY_THRESHOLD:
                roll = best_match['roll_number']
                name = best_match['name']
                conf = round(highest_score, 3)
                matched = True
                overall_match = True
                box_color = (34, 197, 94)
                status_txt = f"{roll} {name} ({conf*100:.1f}%)"
                message = f"Face Matched: {roll} - {name} ({conf*100:.1f}%). Attendance marked Present."

                try:
                    conn = get_db()
                    cur = conn.cursor()
                    today_str = datetime.date.today().isoformat()
                    time_str = datetime.datetime.now().strftime('%H:%M:%S')
                    cur.execute("""
                        INSERT INTO attendance (roll_number, subject, date, time, status, source)
                        VALUES (?, ?, ?, ?, 'Present', 'face_ai')
                        ON CONFLICT(roll_number, subject, date) DO UPDATE SET status='Present', time=excluded.time, source='face_ai'
                    """, (roll, subject, today_str, time_str))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"Attendance mark note: {e}")
            else:
                roll = "UNKNOWN"
                name = "Unrecognized"
                conf = round(highest_score, 3)
                matched = False
                box_color = (59, 59, 239)
                status_txt = f"UNKNOWN ({conf*100:.1f}%)"
                message = "Face unrecognized or no registered match found above similarity threshold."

            landmarks = {
                'left_eye': [int(x + w * 0.32), int(y + h * 0.38)],
                'right_eye': [int(x + w * 0.68), int(y + h * 0.38)],
                'nose': [int(x + w * 0.50), int(y + h * 0.58)],
                'mouth': [int(x + w * 0.50), int(y + h * 0.78)]
            }

            results.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'landmarks': landmarks,
                'roll_number': roll,
                'student_name': name,
                'predicted_identity': f"{roll} - {name}",
                'confidence': conf,
                'confidence_pct': round(conf * 100, 1),
                'match': matched,
                'attendance_marked': matched
            })

            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), box_color, 3)
            cv2.rectangle(annotated_frame, (x, y - 32), (x + w, y), box_color, -1)
            cv2.putText(annotated_frame, status_txt, (x + 6, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    _, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    top_res = results[0] if (overall_match and results) else (results[0] if results else {})
    status_str = 'success' if overall_match else ('identity_mismatch' if any(r.get('is_impersonation') for r in results) else ('mismatch' if results and not overall_match else 'no_face'))
    return {
        'status': status_str,
        'faces_detected': len(faces),
        'match': overall_match,
        'is_impersonation': any(r.get('is_impersonation') for r in results),
        'matched_identity': top_res.get('predicted_identity'),
        'student_name': top_res.get('student_name'),
        'roll_number': top_res.get('roll_number'),
        'confidence': top_res.get('confidence', 0.0),
        'confidence_pct': round(top_res.get('confidence', 0.0) * 100, 1),
        'attendance_marked': any(r.get('attendance_marked') for r in results),
        'results': results,
        'annotated_image': f'data:image/jpeg;base64,{annotated_b64}',
        'model_used': 'Adaptive Haar-Cascade + Multi-Feature Descriptor Matcher',
        'message': message
    }
