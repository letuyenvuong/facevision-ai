# 🧠 CLAUDE CONTEXT MEMORY — FaceVision AI Project

> **Mục đích file này**: Agent đọc file này mỗi khi bắt đầu session mới để nạp lại toàn bộ ngữ cảnh dự án, tránh bắt đầu từ đầu.

---

## 📌 PROJECT OVERVIEW

| Field | Value |
|-------|-------|
| **Tên dự án** | FaceVision AI |
| **Mô tả** | Hệ thống phân tích khuôn mặt realtime toàn diện |
| **Ngôn ngữ** | Python 3.10+ |
| **Framework** | Flask (web server + video stream) |
| **CV Libraries** | OpenCV, DeepFace, NumPy, PIL |
| **Output** | Realtime Camera (webcam + video file) qua browser |
| **Giao diện** | Web App — MJPEG stream qua Flask |

---

## 🏗️ KIẾN TRÚC HỆ THỐNG

```
┌─────────────────────────────────────────────────────────────┐
│                     FaceVision AI Pipeline                  │
│                                                             │
│  [Webcam/Video] ──► [Frame Buffer] ──► [Processing Thread]  │
│                                              │              │
│                              ┌───────────────┤              │
│                              ▼               ▼              │
│                       [Detection]      [FPS Throttle]       │
│                              │                              │
│                    ┌─────────┴──────────┐                   │
│                    ▼                    ▼                   │
│             [Recognition]          [Emotion]                │
│                    │                    │                   │
│                    └─────────┬──────────┘                   │
│                              ▼                              │
│                    [Reconstruction]                         │
│                              │                              │
│                    [Overlay Renderer]                       │
│                              │                              │
│              [Flask MJPEG Stream /video_feed]               │
│                              │                              │
│                    [Browser Frontend UI]                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN

```
facevision-ai/
│
├── app.py                      # Flask entry point, routes, video stream
├── config.py                   # Cấu hình toàn cục (threshold, FPS, model paths)
├── requirements.txt            # Dependencies
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── pipeline.py             # Orchestrator: kết hợp 4 module
│   ├── camera.py               # Camera/video capture + threading
│   └── overlay.py              # Vẽ bounding box, label, emotion bar lên frame
│
├── modules/
│   ├── __init__.py
│   ├── detection.py            # MODULE 1: Face Detection (MTCNN / OpenCV)
│   ├── recognition.py          # MODULE 2: Face Recognition (DeepFace ArcFace)
│   ├── emotion.py              # MODULE 3: Emotion Analysis (DeepFace / FER)
│   └── reconstruction.py       # MODULE 4: Face Reconstruction (PCA Eigenface)
│
├── models/
│   ├── known_faces/            # Ảnh khuôn mặt đã đăng ký (để recognition)
│   ├── embeddings/             # Cache face embeddings (.npy files)
│   └── eigenfaces/             # Dữ liệu PCA reconstruction
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── assets/
│
├── templates/
│   └── index.html              # Giao diện chính
│
├── utils/
│   ├── __init__.py
│   ├── face_db.py              # Quản lý database khuôn mặt đã biết
│   ├── image_utils.py          # Tiện ích xử lý ảnh
│   └── logger.py               # Logging module
│
└── tests/
    ├── test_detection.py
    ├── test_emotion.py
    └── test_recognition.py
```

---

## 🔬 CHI TIẾT 4 MODULE

### MODULE 1 — Detection (`modules/detection.py`)
- **Thuật toán chính**: MTCNN (Multi-task Cascaded CNN) — độ chính xác cao
- **Fallback**: OpenCV Haar Cascade — nhanh hơn khi low resource
- **Output**: List các bounding box `[x, y, w, h]` + confidence score
- **Key function**: `detect_faces(frame) -> List[FaceRegion]`
- **Tối ưu**: Resize frame xuống 50% trước khi detect, scale bbox ngược lại

### MODULE 2 — Recognition (`modules/recognition.py`)
- **Model**: ArcFace via DeepFace (`model_name="ArcFace"`)
- **Similarity metric**: Cosine distance (threshold ≤ 0.4 = match)
- **Database**: Face embeddings lưu trong `models/embeddings/` dạng `.npy`
- **Key functions**:
  - `register_face(name, image_path)` — đăng ký khuôn mặt mới
  - `identify_face(face_crop) -> (name, confidence)` — nhận dạng
- **Tối ưu**: Cache embeddings, không re-compute mỗi frame

### MODULE 3 — Emotion (`modules/emotion.py`)
- **Model**: DeepFace emotion model (CNN-based, 7 classes)
- **Classes**: `angry | disgust | fear | happy | sad | surprise | neutral`
- **Output**: Dict `{emotion: probability}` + dominant emotion
- **Key function**: `analyze_emotion(face_crop) -> EmotionResult`
- **Tối ưu**: Chạy inference mỗi 5 frame, cache kết quả

### MODULE 4 — Reconstruction (`modules/reconstruction.py`)
- **Thuật toán**: PCA Eigenface (Principal Component Analysis)
- **Mục đích**:
  - Reconstruct khuôn mặt từ principal components
  - Hiển thị "face wireframe" / "averaged face"
  - Detect anomaly (khuôn mặt không khớp với distribution)
- **Key functions**:
  - `train_eigenfaces(face_dataset)` — train PCA
  - `reconstruct_face(face_crop) -> reconstructed_image`
  - `get_reconstruction_error(face_crop) -> float` — dùng cho anomaly detection
- **Nâng cao (optional)**: 3DMM lite với dlib 68-landmark

---

## ⚙️ CẤU HÌNH (`config.py`)

```python
# Video
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_LIMIT = 30
PROCESS_EVERY_N_FRAMES = 3   # Chỉ xử lý 1/3 frame để tiết kiệm CPU

# Detection
DETECTION_BACKEND = "mtcnn"   # hoặc "opencv", "retinaface"
MIN_FACE_SIZE = 50             # pixel

# Recognition
RECOGNITION_MODEL = "ArcFace"
RECOGNITION_THRESHOLD = 0.40  # cosine distance
UNKNOWN_LABEL = "Unknown"

# Emotion
EMOTION_PROCESS_EVERY = 5     # frames
EMOTION_ACTIONS = ["emotion"]

# Reconstruction
PCA_COMPONENTS = 50
FACE_SIZE = (64, 64)           # resize trước khi PCA
```

---

## 🔗 FLASK ROUTES

| Route | Method | Mô tả |
|-------|--------|-------|
| `/` | GET | Trang chính (index.html) |
| `/video_feed` | GET | MJPEG stream realtime |
| `/api/register` | POST | Đăng ký khuôn mặt mới (upload ảnh + tên) |
| `/api/faces` | GET | Danh sách khuôn mặt đã đăng ký |
| `/api/status` | GET | FPS, số khuôn mặt detected, trạng thái |
| `/api/snapshot` | GET | Chụp frame hiện tại (base64 PNG) |

---

## 📦 DEPENDENCIES (`requirements.txt`)

```
flask>=2.3.0
opencv-python>=4.8.0
deepface>=0.0.79
numpy>=1.24.0
Pillow>=10.0.0
mtcnn>=0.1.1
tensorflow>=2.13.0   # Backend cho DeepFace
scikit-learn>=1.3.0  # PCA Eigenface
scipy>=1.11.0
fer>=22.5.1          # Fallback emotion (optional)
```

---

## 🎨 FRONTEND

- **Stack**: HTML5 + CSS3 + Vanilla JS (không dùng framework nặng)
- **Layout**: Split panel — trái: live camera stream, phải: realtime stats
- **Hiển thị**:
  - Bounding box màu theo emotion (happy=xanh, angry=đỏ, v.v.)
  - Tên identity + confidence %
  - Emotion bar chart realtime
  - Reconstruction preview thumbnail
  - FPS counter + face count

---

## 🚨 QUAN TRỌNG — CÁC QUYẾT ĐỊNH KỸ THUẬT

1. **Threading**: Camera capture chạy thread riêng, processing chạy thread riêng — tránh blocking Flask
2. **Frame queue**: Dùng `queue.Queue(maxsize=2)` — drop frame cũ nếu processing chậm
3. **Model loading**: Load tất cả model 1 lần lúc khởi động (`app.py` startup), không load lại mỗi request
4. **Eigenface training**: Train offline với dataset, lưu PCA model vào file `.pkl`
5. **Error handling**: Mỗi module có try/except riêng — 1 module lỗi không crash toàn bộ pipeline

---

## 📝 GHI CHÚ PHÁT TRIỂN

- Ngày bắt đầu: _(điền khi bắt đầu code)_
- Developer: _(tên)_
- Python venv: `venv/` (không commit)
- Model weights: tải tự động bởi DeepFace lần đầu chạy (~500MB)
- Test webcam: `python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"`
