"""
Kiem thu toan dien FaceVision AI.
Chay: python tests/test_system.py
"""
import sys, os, time, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import tensorflow  # must be first
import cv2, numpy as np

PASS = "[PASS]"; FAIL = "[FAIL]"; WARN = "[WARN]"; INFO = "[INFO]"
results = []

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    d = f" -- {detail}" if detail else ""
    print(f"  {tag} {name}{d}")
    results.append((name, bool(ok)))
    return bool(ok)

# ────────────────────────────────────────────────────
print("\n--- [1] IMPORTS & CONFIG ---")
try:
    import config
    check("config.BASE_DIR",    os.path.isdir(config.BASE_DIR))
    check("config.CAMERA_SOURCE", True, str(config.CAMERA_SOURCE))
    check("RECOGNITION_THRESHOLD", True, str(config.RECOGNITION_THRESHOLD))
    from utils.face_db import FaceDatabase, FaceRegion, EmotionResult
    from utils.image_utils import crop_face
    check("utils imports", True)
except Exception as e:
    check("imports", False, str(e))

# ────────────────────────────────────────────────────
print("\n--- [2] DEEPFACE WEIGHTS ---")
try:
    from deepface import DeepFace
    weights_dir = os.path.join(os.path.expanduser("~"), ".deepface", "weights")
    arcface_ok  = os.path.exists(os.path.join(weights_dir, "arcface_weights.h5"))
    emotion_ok  = os.path.exists(os.path.join(weights_dir, "facial_expression_model_weights.h5"))
    check("ArcFace weights",  arcface_ok,  os.path.join(weights_dir, "arcface_weights.h5"))
    check("Emotion weights",  emotion_ok,  os.path.join(weights_dir, "facial_expression_model_weights.h5"))
except Exception as e:
    check("DeepFace import", False, str(e))

# ────────────────────────────────────────────────────
print("\n--- [3] BGR vs RGB (ArcFace embedding consistency) ---")
try:
    from deepface import DeepFace
    face_bgr = np.random.randint(80, 180, (112, 112, 3), dtype=np.uint8)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, face_bgr); tmp = f.name
    emb_file = np.array(DeepFace.represent(img_path=tmp, model_name="ArcFace",
        enforce_detection=False, detector_backend="skip")[0]["embedding"], dtype=np.float32)
    os.unlink(tmp)

    emb_bgr = np.array(DeepFace.represent(img_path=face_bgr, model_name="ArcFace",
        enforce_detection=False, detector_backend="skip")[0]["embedding"], dtype=np.float32)
    emb_rgb = np.array(DeepFace.represent(img_path=face_rgb, model_name="ArcFace",
        enforce_detection=False, detector_backend="skip")[0]["embedding"], dtype=np.float32)

    def cos(a, b):
        a = a / (np.linalg.norm(a) + 1e-10)
        b = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a, b))

    s_fb = cos(emb_file, emb_bgr)
    s_fr = cos(emb_file, emb_rgb)
    print(f"  {INFO} file(BGR) vs ndarray(BGR): {s_fb:.6f}")
    print(f"  {INFO} file(BGR) vs ndarray(RGB): {s_fr:.6f}")
    check("BGR matches file path", s_fb > s_fr,
          f"BGR sim={s_fb:.4f} > RGB sim={s_fr:.4f}")
except Exception as e:
    check("BGR/RGB test", False, str(e))

# ────────────────────────────────────────────────────
print("\n--- [4] FACE DETECTION ---")
try:
    from modules.detection import FaceDetector
    detector = FaceDetector()
    check("FaceDetector init", True, "MTCNN" if detector._mtcnn else "Haar")

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        t0 = time.monotonic()
        regions = detector.detect(frame)
        ms = (time.monotonic() - t0) * 1000
        check("detect() on webcam", True, f"{len(regions)} face(s) in {ms:.0f}ms")
        for i, r in enumerate(regions):
            check(f"  region[{i}] valid", r.w > 0 and r.h > 0,
                  f"({r.x},{r.y}) {r.w}x{r.h} conf={r.confidence:.2f}")
    else:
        print(f"  {WARN} No webcam available")
except Exception as e:
    check("Face detection", False, str(e))

# ────────────────────────────────────────────────────
print("\n--- [5] FACE REGISTRATION & IDENTIFICATION ---")
try:
    from modules.recognition import FaceRecognizer
    rec = FaceRecognizer()

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, frame); img_path = f.name

        t0 = time.monotonic()
        saved = rec.register("_test_", img_path, detector=detector)
        elapsed = time.monotonic() - t0
        os.unlink(img_path)
        check("register() with reused detector", os.path.exists(saved),
              f"{os.path.basename(saved)} in {elapsed:.1f}s")

        regions = detector.detect(frame) if "detector" in dir() else []
        if regions:
            crop = crop_face(frame, regions[0].x, regions[0].y,
                             regions[0].w, regions[0].h, padding=0.2)
            name, dist = rec.identify(crop)
            check("identify() same face", name == "_test_",
                  f"name={name!r} dist={dist:.4f} threshold={rec.threshold}")
            check("cosine distance < threshold", dist < rec.threshold,
                  f"dist={dist:.4f} vs threshold={rec.threshold}")
        else:
            print(f"  {WARN} No face detected for identification test")

        rec.db.delete("_test_")
        check("cleanup test data", True)
    else:
        print(f"  {WARN} No webcam available")
except Exception as e:
    check("Registration/Identification", False, str(e))
    import traceback; traceback.print_exc()

# ────────────────────────────────────────────────────
print("\n--- [6] EMOTION ANALYSIS ---")
try:
    from modules.emotion import EmotionAnalyzer, SMOOTH_WINDOW
    check("SMOOTH_WINDOW reduced", SMOOTH_WINDOW <= 3, str(SMOOTH_WINDOW))

    analyzer = EmotionAnalyzer()
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret and "regions" in dir() and regions:
        crop_emo = crop_face(frame, regions[0].x, regions[0].y,
                             regions[0].w, regions[0].h, padding=0.05)
        if crop_emo is not None:
            r = analyzer.analyze(crop_emo, face_id=0)
            check("emotion on tight crop", r.dominant in
                  ["angry","disgust","fear","happy","sad","surprise","neutral"],
                  f"dominant={r.dominant} conf={r.confidence:.2f}")
    else:
        # Test on synthetic face
        dummy = np.ones((64, 64, 3), dtype=np.uint8) * 180
        r = analyzer.analyze(dummy, face_id=0)
        check("emotion on dummy", True, f"dominant={r.dominant}")
except Exception as e:
    check("Emotion analysis", False, str(e))

# ────────────────────────────────────────────────────
print("\n--- [7] CAMERA get_latest() ---")
try:
    from core.camera import CameraStream
    cam = CameraStream(source=0)
    cam.start()
    time.sleep(0.8)

    f1 = cam.get_latest()
    check("get_latest() not None", f1 is not None)
    if f1 is not None:
        check("get_latest() is ndarray", isinstance(f1, np.ndarray), str(f1.shape))

    # Verify non-destructive: call twice, stream still works
    f2 = cam.get_latest()
    f3 = cam.read()
    check("get_latest() non-destructive (read() still works)", f3 is not None)

    cam.stop()
    check("camera stop", not cam.is_running())
except Exception as e:
    check("Camera get_latest", False, str(e))

# ────────────────────────────────────────────────────
print("\n--- [8] PIPELINE (background thread) ---")
try:
    from core.pipeline import FacePipeline, _iou

    r1 = FaceRegion(0, 0, 100, 100)
    r2 = FaceRegion(50, 50, 100, 100)
    r3 = FaceRegion(300, 300, 100, 100)
    check("_iou overlap",    0 < _iou(r1, r2) < 1, f"{_iou(r1,r2):.3f}")
    check("_iou no-overlap", _iou(r1, r3) == 0.0)

    pipe = FacePipeline()
    check("all modules loaded",
          all([pipe._detector, pipe._recognizer, pipe._emotion_analyzer]))

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        annotated, res = pipe.process_frame(frame)
        check("process_frame shape", annotated.shape == frame.shape)
        check("process_frame returns list", isinstance(res, list))
        time.sleep(2.0)   # let inference thread complete at least one run
        status = pipe.get_status()
        check("status has face_count",       "face_count" in status)
        check("status has dominant_emotion", "dominant_emotion" in status)
        print(f"  {INFO} status: {status}")
    else:
        print(f"  {WARN} No webcam")
except Exception as e:
    check("Pipeline", False, str(e))
    import traceback; traceback.print_exc()

# ────────────────────────────────────────────────────
print("\n--- [9] FLASK ROUTES (test client) ---")
try:
    import app as flask_app
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as client:
        r = client.get("/")
        check("GET /",              r.status_code == 200, f"HTTP {r.status_code}")
        r = client.get("/api/faces")
        check("GET /api/faces",     r.status_code == 200, str(r.get_json()))
        r = client.delete("/api/faces/nonexistent_xyz")
        check("DELETE nonexistent", r.status_code == 404, str(r.get_json()))
        r = client.post("/api/register/snapshot",
                        json={}, content_type="application/json")
        check("POST snapshot missing name", r.status_code == 400)
except Exception as e:
    check("Flask routes", False, str(e))

# ────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
passed = sum(1 for _, ok in results if ok)
total  = len(results)
failed = [n for n, ok in results if not ok]
print(f"  {passed}/{total} passed")
if failed:
    print("  Failed:")
    for n in failed:
        print(f"    - {n}")
print()
sys.exit(0 if not failed else 1)
