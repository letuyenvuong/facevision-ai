import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Video / Camera ---
# CAMERA_SOURCE: 0/1/2 for webcam index, or full RTSP URL
#   e.g.  CAMERA_SOURCE=0
#         CAMERA_SOURCE=rtsp://admin:pass@192.168.1.100:554/stream1
_raw = os.environ.get("CAMERA_SOURCE", "0")
try:
    CAMERA_SOURCE = int(_raw)
except ValueError:
    CAMERA_SOURCE = _raw          # keep as string (RTSP / file path)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_LIMIT = 30
# PROCESS_EVERY_N_FRAMES is no longer used — pipeline now uses a
# background inference thread that runs as fast as hardware allows.

# RTSP-specific
RTSP_RECONNECT_DELAY = 3          # seconds before reconnect attempt
RTSP_MAX_RECONNECT_DELAY = 30     # cap for exponential backoff

# --- Detection ---
DETECTION_BACKEND = "mtcnn"   # "mtcnn" | "opencv" | "retinaface"
MIN_FACE_SIZE = 50

# --- Recognition ---
RECOGNITION_MODEL = "ArcFace"
RECOGNITION_THRESHOLD = 0.40
UNKNOWN_LABEL = "Unknown"

# --- Emotion ---
EMOTION_PROCESS_EVERY = 5
EMOTION_ACTIONS = ["emotion"]

# --- Reconstruction ---
PCA_COMPONENTS = 50
FACE_SIZE = (64, 64)

# --- Paths ---
MODELS_DIR = os.path.join(BASE_DIR, "models")
KNOWN_FACES_DIR = os.path.join(MODELS_DIR, "known_faces")
EMBEDDINGS_DIR = os.path.join(MODELS_DIR, "embeddings")
EIGENFACES_DIR = os.path.join(MODELS_DIR, "eigenfaces")
PCA_MODEL_PATH = os.path.join(EIGENFACES_DIR, "pca_model.pkl")

# --- Flask ---
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
SECRET_KEY = "facevision-secret-key"

# --- Emotion color map (BGR for OpenCV) ---
EMOTION_COLORS = {
    "happy":    (136, 255, 0),    # #00FF88
    "angry":    (51,  51,  255),  # #FF3333
    "sad":      (255, 136, 51),   # #3388FF
    "surprise": (0,   170, 255),  # #FFAA00
    "fear":     (255, 0,   170),  # #AA00FF
    "disgust":  (0,   255, 136),  # #88FF00
    "neutral":  (255, 255, 255),  # #FFFFFF
}
DEFAULT_COLOR = (200, 200, 200)
