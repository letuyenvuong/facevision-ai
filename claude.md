# 🧠 CLAUDE CONTEXT MEMORY — FaceVision AI Project

> **Mục đích file này**: Agent đọc file này mỗi khi bắt đầu session mới để nạp lại toàn bộ ngữ cảnh dự án, tránh bắt đầu từ đầu.
> **Cập nhật lần cuối**: 2026-06-09 — Chuyển sang máy GPU mới, bổ sung toàn bộ mục thực nghiệm E1–E4.

---

## 📌 PROJECT OVERVIEW

| Field | Value |
|-------|-------|
| **Tên dự án** | FaceVision AI |
| **Mô tả** | Hệ thống phân tích khuôn mặt realtime toàn diện |
| **Ngôn ngữ** | Python 3.10+ |
| **Framework** | Flask (web server + video stream) |
| **CV Libraries** | OpenCV, DeepFace, NumPy, PIL, scikit-learn |
| **Output** | Realtime Camera (webcam + video file) qua browser |
| **Giao diện** | Web App — MJPEG stream qua Flask |
| **Môi trường hiện tại** | Máy mới có GPU — cần cài lại dependencies |

---

## 🎯 TRẠNG THÁI HIỆN TẠI (quan trọng — đọc trước)

```
✅ Code pipeline realtime: XONG (33/45 checkpoints)
✅ Tất cả 4 module đã viết: detection, recognition, emotion, reconstruction
✅ Flask API + Frontend: XONG
❌ Chưa cài dependencies trên máy mới (chạy pip install trước tiên)
❌ Chưa chạy app lần nào (Phase 9–10 = 0/8)
❌ Chưa có script thực nghiệm E1–E4 (cần viết mới hoàn toàn)
❌ PCA model chưa train (reconstruction tự disable cho đến khi có dataset)
⚠️  DeepFace tải weights ~500MB lần đầu chạy — cần internet
```

**Việc Agent cần làm theo thứ tự:**
1. Cài môi trường trên máy GPU mới
2. Chạy app lần đầu, xác nhận hoạt động (Phase 9)
3. Viết và chạy 4 script thực nghiệm E1–E4 (mục tiêu chính)
4. Xuất kết quả (CSV + PNG) để điền vào báo cáo Word

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

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN (đầy đủ)

```
facevision-ai/
│
├── app.py                      # Flask entry point, routes, video stream ✅
├── config.py                   # Cấu hình toàn cục ✅
├── requirements.txt            # Dependencies ✅
├── README.md
│
├── core/
│   ├── __init__.py             ✅
│   ├── pipeline.py             ✅ FacePipeline orchestrator
│   ├── camera.py               ✅ CameraStream với threading
│   └── overlay.py              ✅ draw_face_box, draw_fps, draw_reconstruction_thumbnail
│
├── modules/
│   ├── __init__.py             ✅
│   ├── detection.py            ✅ FaceDetector — Haar + MTCNN + fallback
│   ├── recognition.py          ✅ FaceRecognizer — ArcFace via DeepFace
│   ├── emotion.py              ✅ EmotionAnalyzer + EmotionTracker (rolling avg)
│   └── reconstruction.py      ✅ FaceReconstructor — PCA Eigenface
│
├── models/
│   ├── known_faces/            # Ảnh khuôn mặt đăng ký
│   ├── embeddings/             # Cache .npy embeddings
│   └── eigenfaces/             # pca_model.pkl (chưa train)
│
├── static/css/style.css        ✅ Dark theme, accent teal #00e5cc
├── static/js/main.js           ✅ Polling, file upload, toast
├── templates/index.html        ✅ Split layout 70/30
│
├── utils/
│   ├── __init__.py             ✅
│   ├── face_db.py              ✅ FaceDatabase — load/save/find_closest cosine
│   ├── image_utils.py          ✅ resize, normalize, crop_face, align_face
│   └── logger.py               ✅ ANSI color logger
│
├── scripts/
│   └── register_face.py        ✅ CLI để đăng ký khuôn mặt
│
├── tests/
│   ├── test_env.py             ✅ Kiểm tra dependencies
│   ├── test_detection.py       ❌ Chưa chạy
│   ├── test_emotion.py         ❌ Chưa chạy
│   └── test_recognition.py     ❌ Chưa chạy
│
└── experiments/                ❌ Chưa tồn tại — Agent cần tạo
    ├── dataset/                # Ảnh thu thập theo điều kiện
    │   └── metadata.csv        # person_id, session, pose, lighting, occlusion
    ├── results/
    │   ├── E1_detector/        # Haar vs MTCNN — Recall, FPS, latency
    │   ├── E2_alignment/       # Cosine similarity có/không alignment
    │   ├── E3_emotion/         # Confusion matrix FER, macro-F1
    │   └── E4_pca/             # MSE/SSIM theo k components
    ├── figures/                # Biểu đồ PNG xuất ra
    ├── run_E1_detector.py      ❌ Cần viết
    ├── run_E2_alignment.py     ❌ Cần viết
    ├── run_E3_emotion.py       ❌ Cần viết
    └── run_E4_pca.py           ❌ Cần viết
```

---

## 🔬 CHI TIẾT 4 MODULE (trạng thái hiện tại)

### MODULE 1 — Detection (`modules/detection.py`) ✅
- Thuật toán chính: MTCNN — resize frame 50%, detect, scale bbox x2
- Fallback: OpenCV Haar Cascade tự động khi MTCNN fail
- Output: `List[FaceRegion(x, y, w, h, confidence, landmarks)]`
- Key function: `FaceDetector.detect(frame) -> List[FaceRegion]`

### MODULE 2 — Recognition (`modules/recognition.py`) ✅
- Model: ArcFace via DeepFace, `detector_backend="skip"` (crop trước)
- Embedding: 512D float32, so sánh cosine distance
- Threshold: ≤ 0.40 = match, > 0.40 = "Unknown"
- Key functions: `extract_embedding(face_crop)`, `identify_face(face_crop)`
- Lưu trữ: `{name}_{timestamp}.npy` trong `models/embeddings/`

### MODULE 3 — Emotion (`modules/emotion.py`) ✅
- Model: DeepFace emotion CNN, 7 classes
- Classes: `angry | disgust | fear | happy | sad | surprise | neutral`
- Smoothing: `EmotionTracker` dùng `deque(maxlen=5)` rolling average
- Key function: `EmotionAnalyzer.analyze(face_crop) -> EmotionResult`
- Chạy inference mỗi 5 frame (config: EMOTION_PROCESS_EVERY)

### MODULE 4 — Reconstruction (`modules/reconstruction.py`) ✅ (chưa train)
- Thuật toán: PCA Eigenface, sklearn, n_components=50
- Lưu model: `models/eigenfaces/pca_model.pkl`
- Key functions: `train(face_images)`, `reconstruct(face_crop)`, `get_reconstruction_error(face_crop)`
- Tự disable nếu pca_model.pkl chưa tồn tại
- Face size chuẩn hóa: (64, 64) grayscale trước PCA

---

## ⚙️ CẤU HÌNH (`config.py`) — các giá trị quan trọng

```python
CAMERA_INDEX = 0
PROCESS_EVERY_N_FRAMES = 3
DETECTION_BACKEND = "mtcnn"   # hoặc "opencv", "retinaface"
MIN_FACE_SIZE = 50
RECOGNITION_MODEL = "ArcFace"
RECOGNITION_THRESHOLD = 0.40
EMOTION_PROCESS_EVERY = 5
PCA_COMPONENTS = 50
FACE_SIZE = (64, 64)
```

---

## 🧪 4 THỰC NGHIỆM CẦN CHẠY CHO BÁO CÁO

Đây là phần quan trọng nhất hiện tại. Kết quả các thực nghiệm này sẽ điền vào Chương 7 của báo cáo Word.

### E1 — So sánh Detector: Haar Cascade vs MTCNN
**File cần tạo**: `experiments/run_E1_detector.py`

**Input**: Thư mục ảnh dataset theo điều kiện (frontal, pose_15, pose_30, dark, occluded)

**Cần đo**:
- Recall = TP / (TP + FN) theo từng điều kiện cho cả 2 detector
- Latency (ms/frame) trung bình cho cả 2 detector
- FPS thực tế

**Output files**:
- `experiments/results/E1_detector/E1_results.csv` — bảng số liệu đầy đủ
- `experiments/results/E1_detector/E1_recall_by_condition.png` — biểu đồ grouped bar
- `experiments/results/E1_detector/E1_latency_boxplot.png` — boxplot latency

**Trả lời câu hỏi nghiên cứu**: RQ4 — MTCNN và Haar khác nhau thế nào về Recall và tốc độ?

---

### E2 — Ảnh hưởng Face Alignment lên Recognition
**File cần tạo**: `experiments/run_E2_alignment.py`

**Input**: Dataset ảnh + embeddings ArcFace

**Cần đo**:
- Tạo cặp genuine (cùng người, khác ảnh) và impostor (khác người)
- Cosine similarity phân bố: 2 trường hợp (không alignment / có 5-point alignment)
- ROC curve, EER, TAR@FAR=0.01 cho cả 2 trường hợp
- Genuine mean ± std và Impostor mean ± std

**Output files**:
- `experiments/results/E2_alignment/scores_no_align.csv`
- `experiments/results/E2_alignment/scores_aligned.csv`
- `experiments/results/E2_alignment/E2_histogram.png` — histogram 2 distributions
- `experiments/results/E2_alignment/E2_roc.png` — ROC curve so sánh
- `experiments/results/E2_alignment/E2_summary.json` — EER, TAR@FAR values

**Trả lời câu hỏi nghiên cứu**: RQ1 — Căn chỉnh ảnh hưởng thế nào đến embedding? RQ2 — Ngưỡng cosine phù hợp?

---

### E3 — FER: Frame-level vs Temporal Smoothing
**File cần tạo**: `experiments/run_E3_emotion.py`

**Input**: Ảnh hoặc video clip với 7 biểu cảm có nhãn ground truth

**Cần đo**:
- Confusion matrix 7×7 chuẩn hóa theo hàng (Recall per class)
- Accuracy toàn bộ + macro-F1 (frame-level)
- F1 từng lớp riêng biệt
- Switch rate (lần/giây) trước và sau EMA smoothing (window=5)
- Latency (ms) với và không có smoothing

**Output files**:
- `experiments/results/E3_emotion/E3_confusion_matrix.png` — heatmap chuẩn hóa
- `experiments/results/E3_emotion/E3_per_class_f1.png` — bar chart F1 từng lớp
- `experiments/results/E3_emotion/E3_results.json` — macro_f1, per_class_f1, switch_rate

**Trả lời câu hỏi nghiên cứu**: RQ3 — Độ chính xác và ổn định FER thay đổi theo điều kiện?

---

### E4 — PCA Reconstruction: Chất lượng theo số components
**File cần tạo**: `experiments/run_E4_pca.py`

**Input**: Dataset ảnh khuôn mặt (dùng 80% train, 20% test)

**Cần đo với k = 10, 25, 50, 100, 200**:
- MSE giữa ảnh gốc và ảnh tái dựng
- SSIM (Structural Similarity Index)
- Explained variance ratio tích lũy
- Thời gian reconstruct (ms/ảnh)

**Output files**:
- `experiments/results/E4_pca/E4_metrics.csv` — bảng k vs MSE vs SSIM
- `experiments/results/E4_pca/E4_curve_mse_ssim.png` — đường cong k–MSE và k–SSIM
- `experiments/results/E4_pca/E4_scree_plot.png` — explained variance ratio
- `experiments/results/E4_pca/E4_reconstruction_grid.png` — lưới ảnh gốc/tái dựng (5 cột)

**Trả lời câu hỏi nghiên cứu**: RQ5 — PCA khác autoencoder thế nào về độ trung thực?

---

## 📋 FORMAT DỮ LIỆU THU THẬP

### Cấu trúc thư mục dataset
```
experiments/dataset/
├── metadata.csv               # Thông tin từng ảnh
├── person_01/
│   ├── frontal_001.jpg        # Chính diện, ánh sáng đều
│   ├── frontal_002.jpg
│   ├── pose15_001.jpg         # Nghiêng ~15°
│   ├── pose30_001.jpg         # Nghiêng ~30°
│   ├── dark_001.jpg           # Thiếu sáng
│   └── occluded_001.jpg       # Khẩu trang hoặc kính
├── person_02/
│   └── ...
```

### Format metadata.csv
```
person_id,image_file,session,pose,lighting,occlusion,camera,resolution,note
person_01,frontal_001.jpg,1,0deg,even,none,webcam,1280x720,
person_01,pose15_001.jpg,1,15deg,even,none,webcam,1280x720,
person_01,dark_001.jpg,1,0deg,low,none,webcam,1280x720,
```

### Yêu cầu dataset tối thiểu
- Số người: 5–10 (lý tưởng 10–15)
- Điều kiện mỗi người: 5 (frontal, pose_15, pose_30, dark, occluded)
- Ảnh mỗi điều kiện: 5–10 ảnh
- Tổng tối thiểu: ~250 ảnh

---

## 🔗 FLASK ROUTES (đã implement)

| Route | Method | Trạng thái |
|-------|--------|-----------|
| `/` | GET | ✅ index.html |
| `/video_feed` | GET | ✅ MJPEG stream |
| `/api/register` | POST | ✅ Upload ảnh + tên |
| `/api/faces` | GET | ✅ Danh sách registered |
| `/api/status` | GET | ✅ FPS, face_count, uptime |
| `/api/snapshot` | GET | ✅ base64 PNG |

---

## 📦 DEPENDENCIES

```
flask>=2.3.0
opencv-python>=4.8.0
deepface>=0.0.79
numpy>=1.24.0
Pillow>=10.0.0
mtcnn>=0.1.1
tensorflow>=2.13.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0      # Cần thêm cho vẽ biểu đồ thực nghiệm
seaborn>=0.12.0        # Cần thêm cho confusion matrix heatmap
pandas>=2.0.0          # Cần thêm cho xử lý CSV kết quả
fer>=22.5.1            # Fallback emotion (optional)
```

> ⚠️ matplotlib, seaborn, pandas có thể chưa có trong requirements.txt gốc — Agent cần kiểm tra và thêm nếu thiếu.

---

## 🚨 QUAN TRỌNG — CÁC QUYẾT ĐỊNH KỸ THUẬT ĐÃ CHỐT

1. **Threading**: Camera capture thread riêng, processing thread riêng
2. **Frame queue**: `queue.Queue(maxsize=2)` — drop frame cũ khi processing chậm
3. **Model loading**: Load 1 lần lúc khởi động app.py, không lazy load
4. **Eigenface training**: Train offline, lưu `models/eigenfaces/pca_model.pkl`
5. **Error handling**: Mỗi module try/except riêng, 1 module lỗi không crash pipeline
6. **Cosine similarity**: Dùng L2-normalized embeddings, tính dot product
7. **ArcFace input**: Dùng `detector_backend="skip"` — crop face trước khi đưa vào model

---

## 📝 GHI CHÚ MÁY MỚI (GPU)

- Clone: `git clone https://github.com/letuyenvuong/facevision-ai.git`
- Venv: `python -m venv venv && source venv/bin/activate`
- Cài: `pip install -r requirements.txt`
- Thêm: `pip install matplotlib seaborn pandas` (nếu chưa có trong requirements.txt)
- Test webcam: `python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"`
- Chạy app: `python app.py` → mở http://localhost:5000
- DeepFace sẽ tải weights ~500MB lần đầu — cần kết nối internet
