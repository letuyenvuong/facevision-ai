import os
import sys
import time
import uuid
import base64
import threading

# Force UTF-8 stdout so DeepFace emoji logs don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, SECRET_KEY, CAMERA_SOURCE, BASE_DIR
from core.camera import CameraStream
from core.pipeline import FacePipeline
from utils.logger import get_logger

logger = get_logger("app")

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)

_UPLOAD_DIR = os.path.join(BASE_DIR, "models", "known_faces")

# ── Global state ─────────────────────────────────────────────────────
camera: CameraStream | None = None
pipeline: FacePipeline | None = None
_camera_lock  = threading.Lock()
_start_time   = time.time()

# Auto-capture jobs: {job_id: dict}
_capture_jobs: dict = {}


# ── Startup helpers ──────────────────────────────────────────────────

def _start_camera(source) -> CameraStream:
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cam = CameraStream(source=source)
    cam.start()
    return cam


def get_pipeline() -> FacePipeline:
    global camera, pipeline
    with _camera_lock:
        if pipeline is None:
            logger.info("Initializing pipeline...")
            camera = _start_camera(CAMERA_SOURCE)
            pipeline = FacePipeline()
            logger.info("Pipeline ready")
    return pipeline


def generate_frames():
    pipe = get_pipeline()
    while True:
        with _camera_lock:
            cam = camera
        if cam is None or not cam.is_running():
            time.sleep(0.05)
            continue
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        annotated, _ = pipe.process_frame(frame)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# ── Core routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    pipe = get_pipeline()
    status = pipe.get_status()
    status["uptime"] = round(time.time() - _start_time, 1)
    status["camera"] = camera.info() if camera else {"running": False}
    return jsonify(status)


@app.route("/api/camera")
def api_camera():
    if camera is None:
        get_pipeline()
    return jsonify(camera.info() if camera else {"running": False})


@app.route("/api/camera/switch", methods=["POST"])
def api_camera_switch():
    global camera
    data   = request.get_json(silent=True) or {}
    raw    = data.get("source", "").strip()
    if not raw:
        return jsonify({"error": "Missing 'source' field"}), 400
    source = int(raw) if raw.isdigit() else raw
    with _camera_lock:
        old_cam = camera
        try:
            new_cam = _start_camera(source)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        camera = new_cam
    if old_cam:
        threading.Thread(target=old_cam.stop, daemon=True).start()
    logger.info(f"Camera switched to: {source}")
    return jsonify({"success": True, **camera.info()})


# ── Face management ──────────────────────────────────────────────────

@app.route("/api/faces")
def api_faces():
    pipe  = get_pipeline()
    stats = pipe._recognizer.db.get_stats() if pipe._recognizer else {}
    names = sorted(stats.keys())
    return jsonify({
        "faces": names,
        "count": len(names),
        "stats": stats,   # {name: embedding_count}
    })


@app.route("/api/faces/<name>", methods=["DELETE"])
def api_delete_face(name: str):
    pipe = get_pipeline()
    if pipe._recognizer is None:
        return jsonify({"error": "Recognition module not available"}), 503
    deleted = pipe._recognizer.db.delete(name)
    if deleted == 0:
        return jsonify({"error": f"No face found: '{name}'"}), 404
    return jsonify({"success": True, "name": name, "deleted": deleted})


# ── Registration: upload file ────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def api_register():
    if "image" not in request.files or "name" not in request.form:
        return jsonify({"error": "Missing 'image' or 'name'"}), 400
    name = request.form["name"].strip()
    if not name:
        return jsonify({"error": "Name cannot be empty"}), 400

    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    filename  = f"{name}_{int(time.time())}.jpg"
    save_path = os.path.join(_UPLOAD_DIR, filename)
    request.files["image"].save(save_path)

    pipe = get_pipeline()
    if pipe._recognizer is None:
        return jsonify({"error": "Recognition module not available"}), 503
    try:
        pipe._recognizer.register(name, save_path, detector=None)
        stats = pipe._recognizer.db.get_stats()
        return jsonify({"success": True, "name": name,
                        "embeddings": stats.get(name, 0)})
    except Exception as e:
        logger.error(f"Register (upload) failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Registration: single snapshot ───────────────────────────────────

@app.route("/api/register/snapshot", methods=["POST"])
def api_register_snapshot():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Missing 'name' field"}), 400

    pipe = get_pipeline()
    if camera is None:
        return jsonify({"error": "Camera not started"}), 503
    frame = camera.get_latest()
    if frame is None:
        return jsonify({"error": "No frame available yet"}), 503

    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(_UPLOAD_DIR, f"{name}_{int(time.time())}.jpg")
    cv2.imwrite(save_path, frame)

    if pipe._recognizer is None:
        return jsonify({"error": "Recognition module not available"}), 503
    try:
        pipe._recognizer.register(name, save_path, detector=None)
        stats = pipe._recognizer.db.get_stats()
        return jsonify({"success": True, "name": name,
                        "embeddings": stats.get(name, 0)})
    except Exception as e:
        logger.error(f"Register (snapshot) failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Registration: auto-capture (multi-frame) ─────────────────────────

@app.route("/api/register/auto", methods=["POST"])
def api_register_auto():
    """Start a background auto-capture job. Returns job_id to poll."""
    data  = request.get_json(silent=True) or {}
    name  = data.get("name", "").strip()
    count = max(3, min(int(data.get("count", 8)), 15))   # clamp 3-15
    if not name:
        return jsonify({"error": "Missing 'name' field"}), 400

    pipe = get_pipeline()
    if camera is None or pipe._recognizer is None:
        return jsonify({"error": "Pipeline not ready"}), 503

    job_id = uuid.uuid4().hex[:8]
    _capture_jobs[job_id] = {
        "name":     name,
        "status":   "running",
        "progress": 0,
        "total":    count,
        "saved":    0,
        "message":  "Starting...",
    }

    threading.Thread(
        target=_auto_capture_worker,
        args=(job_id, name, count),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id, "name": name, "total": count})


@app.route("/api/register/auto/<job_id>")
def api_register_auto_status(job_id: str):
    job = _capture_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


def _auto_capture_worker(job_id: str, name: str, target: int) -> None:
    """
    Background thread:
      1. Detect face in latest frame
      2. Quality-check the crop
      3. Ensure diversity from previous captures
      4. Extract embedding + register with augmentation
      5. Repeat until target count reached
    """
    from utils.image_utils import crop_face
    from utils.face_quality import (score_quality, quality_label,
                                    cosine_distance, is_diverse_enough)

    job          = _capture_jobs[job_id]
    pipe         = pipeline          # captured at job-start
    cam          = camera
    captured_emb: list = []          # embeddings already accepted
    captured_crops: list = []        # crops already accepted
    max_attempts = target * 10
    attempts     = 0

    while len(captured_crops) < target and attempts < max_attempts:
        attempts += 1
        time.sleep(0.4)              # pace between attempts

        frame = cam.get_latest() if cam else None
        if frame is None:
            job["message"] = "Waiting for camera..."
            continue

        # Detect face (uses locked Haar detector)
        try:
            regions = pipe._detector.detect(frame) if pipe._detector else []
        except Exception:
            regions = []

        if not regions:
            job["message"] = "No face detected — look at the camera"
            continue

        best   = max(regions, key=lambda r: r.w * r.h)
        crop   = crop_face(frame, best.x, best.y, best.w, best.h, padding=0.1)
        if crop is None:
            continue

        # Quality check
        q = score_quality(crop)
        if q["overall"] < 0.30:
            label = quality_label(q["overall"])
            hints = []
            if q["sharpness"]  < 0.40: hints.append("hold still")
            if q["brightness"] < 0.40: hints.append("improve lighting")
            if q["contrast"]   < 0.35: hints.append("face camera directly")
            job["message"] = (f"Quality {label} ({q['overall']:.0%})"
                              + (f" — {', '.join(hints)}" if hints else ""))
            continue

        # Extract embedding for diversity check (cheap, no augmentation yet)
        try:
            emb = pipe._recognizer.extract_embedding(crop)
        except Exception as e:
            job["message"] = f"Embedding error: {e}"
            continue

        # Diversity check — require enough variation from previous captures
        if not is_diverse_enough(emb, captured_emb, min_dist=0.08):
            job["message"] = (f"Frame {len(captured_crops)+1}/{target} — "
                              "move slightly for more variation")
            continue

        # Accept this frame: augment and save all variants
        try:
            saved_n = pipe._recognizer.register_from_crop(name, crop, augment=True)
        except Exception as e:
            job["message"] = f"Save error: {e}"
            continue

        captured_crops.append(crop)
        captured_emb.append(emb)
        job["progress"] = len(captured_crops)
        job["saved"]    = job["saved"] + saved_n
        job["message"]  = (f"Captured {len(captured_crops)}/{target} "
                           f"(+{saved_n} embeddings)")

    # Finished
    if captured_crops:
        job["status"]  = "done"
        job["message"] = (f"Done: {len(captured_crops)} frames, "
                          f"{job['saved']} embeddings saved")
        logger.info(f"Auto-capture '{name}': {job['saved']} embeddings "
                    f"from {len(captured_crops)} frames")
    else:
        job["status"]  = "failed"
        job["message"] = "Could not capture any valid frames"
        logger.warning(f"Auto-capture '{name}': failed after {attempts} attempts")


# ── Face quality (real-time) ─────────────────────────────────────────

@app.route("/api/face/quality")
def api_face_quality():
    """Return quality score for the best face in the current frame."""
    from utils.image_utils import crop_face
    from utils.face_quality import score_quality, quality_label

    pipe  = get_pipeline()
    frame = camera.get_latest() if camera else None
    if frame is None:
        return jsonify({"error": "No frame"}), 503

    try:
        regions = pipe._detector.detect(frame) if pipe._detector else []
    except Exception:
        regions = []

    if not regions:
        return jsonify({"face": False, "message": "No face detected"})

    best = max(regions, key=lambda r: r.w * r.h)
    crop = crop_face(frame, best.x, best.y, best.w, best.h, padding=0.1)
    if crop is None:
        return jsonify({"face": False, "message": "Crop failed"})

    q = score_quality(crop)
    return jsonify({
        "face":    True,
        "quality": q,
        "label":   quality_label(q["overall"]),
        "size":    {"w": best.w, "h": best.h},
    })


# ── Snapshot ─────────────────────────────────────────────────────────

@app.route("/api/snapshot")
def api_snapshot():
    if camera is None:
        return jsonify({"error": "Camera not started"}), 503
    frame = camera.get_latest()
    if frame is None:
        return jsonify({"error": "No frame available"}), 503
    _, buf = cv2.imencode(".png", frame)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return jsonify({"image": f"data:image/png;base64,{b64}"})


# ── Lifecycle ────────────────────────────────────────────────────────

def _on_exit():
    global camera
    if camera:
        camera.stop()


import atexit
atexit.register(_on_exit)

if __name__ == "__main__":
    logger.info(f"Starting FaceVision AI on http://{FLASK_HOST}:{FLASK_PORT}")
    logger.info(f"Camera: {CAMERA_SOURCE!r}  (override: env CAMERA_SOURCE=...)")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, threaded=True)
