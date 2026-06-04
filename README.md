# FaceVision AI

Hệ thống phân tích khuôn mặt realtime chạy trên trình duyệt.

| Tính năng | Chi tiết |
|-----------|---------|
| **Detection** | MTCNN (cao) / Haar Cascade (nhanh) |
| **Recognition** | ArcFace via DeepFace, cosine distance |
| **Emotion** | 7 cảm xúc: angry, disgust, fear, happy, sad, surprise, neutral |
| **Stream** | MJPEG qua Flask, webcam hoặc RTSP |
| **Registration** | Auto-capture 5–12 frames × 6 augmentations = tới 72 embeddings/người |

---

## Yêu cầu

- Python **3.10** hoặc mới hơn
- Webcam hoặc camera RTSP
- RAM ≥ 4 GB (TensorFlow + MTCNN)
- Kết nối internet lần đầu chạy (tải model weights ~150 MB)

---

## Cài đặt

```bash
# 1. Clone
git clone https://github.com/letuyenvuong/facevision-ai.git
cd facevision-ai

# 2. Tạo virtual environment (khuyến nghị)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Cài dependencies
pip install -r requirements.txt
```

---

## Chạy

```bash
# Windows
set PYTHONUTF8=1 && python app.py

# macOS / Linux
PYTHONUTF8=1 python app.py
```

Mở trình duyệt: **http://localhost:5000**

> **Lần đầu chạy:** Khi nhấn "Register" lần đầu, hệ thống sẽ tải ArcFace weights (~137 MB)
> và Emotion weights (~6 MB) về `~/.deepface/weights/`. Việc này chỉ xảy ra **một lần**.

---

## Cấu hình camera

Mặc định dùng webcam index `0`. Thay đổi bằng biến môi trường:

```bash
# Webcam khác
set CAMERA_SOURCE=1 && python app.py

# Camera RTSP (IP camera, NVR, v.v.)
set CAMERA_SOURCE=rtsp://admin:password@192.168.1.100:554/stream1 && python app.py
```

Hoặc đổi trực tiếp trên UI: nhập vào ô **Camera → Switch**.

---

## Đăng ký khuôn mặt

### Cách 1 — Auto Capture (khuyến nghị)

1. Mở tab **Auto Capture**
2. Nhập tên → chọn số frame (5 / 8 / 12)
3. Nhìn thẳng vào camera → nhấn **Start Capture**
4. Xoay đầu nhẹ giữa các lần chụp theo hướng dẫn trên màn hình
5. Hệ thống tự động: kiểm tra chất lượng → lấy embedding → augment 6 biến thể

### Cách 2 — Upload ảnh

1. Tab **Upload Photo** → chọn ảnh chân dung rõ mặt
2. Hệ thống tự detect khuôn mặt và tạo 6 embedding từ 1 ảnh

### Cách 3 — CLI

```bash
python scripts/register_face.py --name "Ten" --image path/to/photo.jpg
```

---

## Cấu trúc thư mục

```
facevision-ai/
├── app.py                  # Flask entry point
├── config.py               # Cấu hình toàn cục
├── requirements.txt
│
├── core/
│   ├── camera.py           # Camera stream + threading
│   ├── overlay.py          # Vẽ bbox / emotion bar lên frame
│   └── pipeline.py         # Orchestrator: detect → recognize → emotion
│
├── modules/
│   ├── detection.py        # MTCNN / Haar face detection
│   ├── recognition.py      # ArcFace embedding + identification
│   ├── emotion.py          # DeepFace emotion analysis
│   └── reconstruction.py   # PCA Eigenface (disabled until trained)
│
├── utils/
│   ├── face_db.py          # Embedding database (load/save/match)
│   ├── face_quality.py     # Quality scoring + augmentation
│   ├── image_utils.py      # Crop, resize, align helpers
│   └── logger.py           # Colour console logger
│
├── models/                 # Tạo tự động khi chạy, không commit
│   ├── embeddings/         # *.npy — face embeddings đã đăng ký
│   ├── known_faces/        # Ảnh gốc lưu khi đăng ký
│   └── eigenfaces/         # PCA model (nếu đã train)
│
├── static/
│   ├── css/style.css
│   └── js/main.js
├── templates/index.html
├── scripts/register_face.py
└── tests/
    ├── test_env.py         # Kiểm tra dependencies
    └── test_system.py      # Kiểm thử toàn hệ thống
```

---

## API

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/video_feed` | MJPEG stream |
| GET | `/api/status` | FPS, face count, emotion, modules |
| GET | `/api/faces` | Danh sách + số embedding mỗi người |
| DELETE | `/api/faces/<name>` | Xoá người khỏi database |
| POST | `/api/register` | Đăng ký từ file ảnh upload |
| POST | `/api/register/snapshot` | Đăng ký từ frame hiện tại |
| POST | `/api/register/auto` | Bắt đầu auto-capture job |
| GET | `/api/register/auto/<id>` | Trạng thái job auto-capture |
| GET | `/api/face/quality` | Điểm chất lượng khuôn mặt trong frame |
| GET | `/api/camera` | Thông tin camera hiện tại |
| POST | `/api/camera/switch` | Đổi nguồn camera (webcam / RTSP) |

---

## Kiểm tra môi trường

```bash
python tests/test_env.py      # Kiểm tra dependencies
python tests/test_system.py   # Kiểm thử toàn bộ pipeline (cần webcam)
```

---

## Lưu ý

- `models/embeddings/` và `models/known_faces/` **không được commit** (dữ liệu cá nhân).
  Sao chép thủ công nếu cần chuyển sang máy khác.
- DeepFace weights tải về `~/.deepface/weights/` — không cần làm gì thêm.
- Trên Windows nên dùng `PYTHONUTF8=1` để tránh lỗi encoding từ DeepFace logger.
