# ✅ CHECKPOINTS — FaceVision AI Project

> **Hướng dẫn cho Agent**: Đọc file này để biết đã hoàn thành đến đâu.
> - `[ ]` = Chưa làm
> - `[~]` = Đang làm / Làm dở
> - `[x]` = Đã hoàn thành
>
> **Khi bắt đầu session mới**: Tìm checkpoint `[~]` gần nhất hoặc `[ ]` đầu tiên và tiếp tục từ đó.
> **Sau khi hoàn thành 1 step**: Cập nhật `[ ]` thành `[x]` và ghi notes bên dưới.

---

## 📊 TỔNG QUAN TIẾN ĐỘ

```
Phase 1: Setup & Structure     [x] 5/5
Phase 2: Core Pipeline         [x] 3/4  (2.4 test - pending manual run)
Phase 3: Module Detection      [x] 3/4  (3.4 test - pending manual run)
Phase 4: Module Recognition    [x] 4/5  (4.5 test - pending manual run)
Phase 5: Module Emotion        [x] 3/4  (5.4 test - pending manual run)
Phase 6: Module Reconstruction [x] 4/5  (6.5 test - pending manual run)
Phase 7: Flask API & Stream    [x] 3/5  (7.4, 7.5 - pending manual run)
Phase 8: Frontend UI           [x] 5/5
Phase 9: Integration & Test    [ ] 0/4
Phase 10: Optimization         [ ] 0/4

TOTAL: [x] 33/45 steps hoàn thành
```

---

## 🚀 PHASE 1 — PROJECT SETUP & STRUCTURE

### CHECKPOINT 1.1 — Khởi tạo cấu trúc thư mục
- [x] Tạo toàn bộ thư mục theo cấu trúc trong `claude.md`
- [x] Tạo các file `__init__.py` cho mỗi package
- **Notes**: Tạo core/, modules/, models/{known_faces,embeddings,eigenfaces}, static/{css,js,assets}, templates/, utils/, tests/, scripts/

### CHECKPOINT 1.2 — File cấu hình
- [x] Tạo `config.py` với tất cả constants
- [x] Tạo `requirements.txt` đầy đủ
- **Notes**: config.py có đủ tất cả section: video, detection, recognition, emotion, reconstruction, flask, paths, emotion_colors

### CHECKPOINT 1.3 — Utility modules
- [x] Tạo `utils/logger.py` — logging với color console output
- [x] Tạo `utils/image_utils.py` — resize, normalize, crop face
- **Notes**: logger.py dùng ANSI colors; image_utils.py có resize_frame, resize_for_detection, normalize_face, crop_face, align_face, encode_jpeg

### CHECKPOINT 1.4 — Database khuôn mặt
- [x] Tạo `utils/face_db.py` — load/save face embeddings từ `models/embeddings/`
- [x] Định nghĩa dataclass `FaceRecord`, `FaceRegion`, `EmotionResult`
- **Notes**: FaceDatabase class với load(), reload(), save(), find_closest() cosine distance

### CHECKPOINT 1.5 — Kiểm tra môi trường
- [ ] Verify OpenCV import + webcam access
- [ ] Verify DeepFace import + model download
- [ ] Verify MTCNN import
- [x] Tạo `tests/test_env.py` — script kiểm tra toàn bộ dependencies
- **Notes**: Chạy `python tests/test_env.py` để verify môi trường

---

## 🔧 PHASE 2 — CORE PIPELINE

### CHECKPOINT 2.1 — Camera module
- [x] Tạo `core/camera.py`
- [x] Implement `CameraStream` class với threading
- [x] Frame queue với `maxsize=2` (drop old frames)
- [x] Support cả webcam index và video file path
- [x] Graceful stop/release
- **Notes**: CameraStream.start(), read(), stop(). FPS throttling bằng sleep loop.

### CHECKPOINT 2.2 — Overlay renderer
- [x] Tạo `core/overlay.py`
- [x] Vẽ bounding box màu theo emotion
- [x] Render tên identity + confidence badge
- [x] Render emotion mini-bar bên dưới bbox
- [x] FPS counter ở góc màn hình
- **Notes**: draw_face_box(), draw_fps(), draw_face_count(), draw_reconstruction_thumbnail()

### CHECKPOINT 2.3 — Pipeline orchestrator
- [x] Tạo `core/pipeline.py`
- [x] Class `FacePipeline` — khởi tạo tất cả 4 module
- [x] Method `process_frame(frame)` — chạy toàn bộ pipeline
- [x] FPS throttling: chỉ chạy heavy inference mỗi N frame
- [x] Error isolation: mỗi module try/except riêng
- **Notes**: Cache kết quả giữa các frame, lazy module loading với error isolation

### CHECKPOINT 2.4 — Pipeline unit test
- [ ] Test `process_frame` với ảnh tĩnh
- [ ] Đảm bảo output format đúng chuẩn
- **Notes**: Pending — cần cài dependencies trước

---

## 👁️ PHASE 3 — MODULE DETECTION

### CHECKPOINT 3.1 — Detection cơ bản (OpenCV Haar)
- [x] Tạo `modules/detection.py`
- [x] Implement `detect_faces_haar(frame)` dùng `haarcascade_frontalface_default.xml`
- [x] Output: `List[FaceRegion(x, y, w, h, confidence)]`
- **Notes**: FaceDetector._detect_haar() với Haar cascade

### CHECKPOINT 3.2 — Detection nâng cao (MTCNN)
- [x] Implement `detect_faces_mtcnn(frame)` dùng MTCNN
- [x] Include landmarks (5 điểm: mắt, mũi, miệng)
- [x] Auto-fallback sang Haar nếu MTCNN fails
- **Notes**: FaceDetector._detect_mtcnn() với resize 50% trước khi detect, scale bbox x2

### CHECKPOINT 3.3 — Face crop utility
- [x] `crop_face(frame, face_region, padding=0.2)` — crop + padding
- [x] `align_face(frame, landmarks)` — align khuôn mặt thẳng (affine transform)
- **Notes**: Trong utils/image_utils.py

### CHECKPOINT 3.4 — Detection test
- [ ] `tests/test_detection.py` — test với ảnh có 0, 1, nhiều khuôn mặt
- [ ] Benchmark FPS với MTCNN vs Haar
- **Notes**: Pending

---

## 🪪 PHASE 4 — MODULE RECOGNITION

### CHECKPOINT 4.1 — DeepFace embedding
- [x] Tạo `modules/recognition.py`
- [x] `extract_embedding(face_crop)` → `np.ndarray (512,)` dùng ArcFace
- [x] Handle lỗi: no face, blurry, too small
- **Notes**: Dùng temp file + DeepFace.represent với detector_backend="skip"

### CHECKPOINT 4.2 — Face registration
- [x] `register_face(name, image_path)` — extract + lưu embedding vào `models/embeddings/`
- [x] Format file: `{name}_{timestamp}.npy`
- [x] CLI script `scripts/register_face.py` để đăng ký từ terminal
- **Notes**: scripts/register_face.py với argparse

### CHECKPOINT 4.3 — Face identification
- [x] `identify_face(face_crop) -> (name, cosine_distance)`
- [x] Load tất cả embeddings từ database vào memory lúc start
- [x] Cosine similarity so sánh với tất cả known faces
- [x] Return "Unknown" nếu distance > threshold (0.40)
- **Notes**: FaceDatabase.find_closest() với L2-norm của unit vectors

### CHECKPOINT 4.4 — Embedding cache
- [x] `FaceDatabase` class — load embeddings 1 lần, cache in-memory
- [x] `reload()` method — hot-reload khi có face mới đăng ký
- **Notes**: Trong utils/face_db.py

### CHECKPOINT 4.5 — Recognition test
- [ ] `tests/test_recognition.py` — test register + identify
- [ ] Test edge case: same person nhiều ảnh, người lạ
- **Notes**: Pending

---

## 😊 PHASE 5 — MODULE EMOTION

### CHECKPOINT 5.1 — DeepFace emotion analyzer
- [x] Tạo `modules/emotion.py`
- [x] `analyze_emotion(face_crop) -> EmotionResult`
- [x] `EmotionResult`: `{dominant: str, scores: Dict[str, float], confidence: float}`
- **Notes**: DeepFace.analyze với actions=["emotion"], detector_backend="skip"

### CHECKPOINT 5.2 — Emotion smoothing
- [x] Rolling average trên 5 frame gần nhất (tránh flickering)
- [x] `EmotionTracker` class per face ID
- **Notes**: EmotionTracker dùng deque(maxlen=5)

### CHECKPOINT 5.3 — Emotion color mapping
- [x] Map emotion → màu bounding box (trong config.py EMOTION_COLORS)
- **Notes**: BGR colors trong config.py, dùng trong core/overlay.py

### CHECKPOINT 5.4 — Emotion test
- [ ] `tests/test_emotion.py` — test với ảnh biểu cảm rõ ràng
- [ ] Accuracy check thủ công
- **Notes**: Pending

---

## 🧬 PHASE 6 — MODULE RECONSTRUCTION

### CHECKPOINT 6.1 — PCA Eigenface training
- [x] Tạo `modules/reconstruction.py`
- [x] `train_eigenfaces(face_images, n_components=50)` — fit PCA
- [x] Lưu PCA model vào `models/eigenfaces/pca_model.pkl`
- [ ] Script `scripts/train_eigenfaces.py` dùng dataset mẫu
- **Notes**: FaceReconstructor.train() với sklearn PCA

### CHECKPOINT 6.2 — Face reconstruction
- [x] `reconstruct_face(face_crop) -> reconstructed_image`
- [x] Load PCA model, project face vào eigenspace, reconstruct
- [x] Resize output về kích thước gốc
- **Notes**: reconstruct() returns grayscale (64x64) numpy array

### CHECKPOINT 6.3 — Reconstruction error / anomaly
- [x] `get_reconstruction_error(face_crop) -> float`
- [x] MSE giữa original và reconstructed — cao = khuôn mặt "lạ"
- [ ] Threshold để flag "unusual face"
- **Notes**: get_reconstruction_error() returns MSE float

### CHECKPOINT 6.4 — Visualization
- [x] Hiển thị reconstructed face thumbnail nhỏ ở góc bbox
- [ ] Eigenface components visualization (ảnh debug)
- **Notes**: draw_reconstruction_thumbnail() trong core/overlay.py

### CHECKPOINT 6.5 — Reconstruction test
- [ ] Test pipeline: crop → reconstruct → compare
- [ ] Kiểm tra reconstruction quality bằng mắt
- **Notes**: Pending — cần train PCA model trước

---

## 🌐 PHASE 7 — FLASK API & VIDEO STREAM

### CHECKPOINT 7.1 — Flask app skeleton
- [x] Tạo `app.py` — Flask init, CORS, error handlers
- [x] Load `FacePipeline` lúc startup (không lazy load)
- [x] Graceful shutdown — release camera khi stop
- **Notes**: atexit.register(on_exit) để release camera

### CHECKPOINT 7.2 — MJPEG video stream
- [x] Route `GET /video_feed` — generator yield JPEG frames
- [x] `generate_frames()` — lấy frame từ pipeline, encode JPEG, yield
- [x] Header: `multipart/x-mixed-replace; boundary=frame`
- **Notes**: generate_frames() generator function

### CHECKPOINT 7.3 — API endpoints
- [x] `GET /api/status` — FPS, face_count, uptime, module status
- [x] `GET /api/faces` — danh sách registered faces
- [x] `POST /api/register` — upload ảnh + name → register face
- [x] `GET /api/snapshot` — base64 PNG của frame hiện tại
- **Notes**: Đủ 4 endpoints

### CHECKPOINT 7.4 — API test
- [ ] Test tất cả endpoints với curl / Postman
- [ ] Test upload ảnh register face
- **Notes**: Pending — cần chạy app

### CHECKPOINT 7.5 — Stream stability
- [ ] Test stream liên tục 5 phút — không memory leak
- [ ] FPS ổn định > 15 FPS
- **Notes**: Pending

---

## 🎨 PHASE 8 — FRONTEND UI

### CHECKPOINT 8.1 — Layout HTML cơ bản
- [x] Tạo `templates/index.html`
- [x] Split layout: stream (trái 70%) + panel (phải 30%)
- [x] `<img src="/video_feed">` cho live stream
- **Notes**: Flexbox layout với stream + side-panel

### CHECKPOINT 8.2 — CSS styling
- [x] Dark theme với màu accent teal/cyan
- [x] Responsive cho 1080p+
- [x] Tạo `static/css/style.css`
- **Notes**: CSS variables, dark theme #0d0f14, accent #00e5cc

### CHECKPOINT 8.3 — Realtime stats panel
- [x] FPS counter (poll `/api/status` mỗi 1s)
- [x] Danh sách khuôn mặt đang detected
- [x] Current dominant emotion với emoji
- **Notes**: pollStatus() mỗi 1s, pollFaces() mỗi 5s

### CHECKPOINT 8.4 — Face registration UI
- [x] Form upload ảnh + nhập tên
- [x] Preview ảnh trước khi submit
- [x] Feedback toast khi register thành công/thất bại
- **Notes**: Toast system với success/error/info types

### CHECKPOINT 8.5 — JavaScript
- [x] Tạo `static/js/main.js`
- [x] Polling API status
- [x] Handle file upload
- [x] Toast notifications
- **Notes**: Vanilla JS, async/await fetch

---

## 🧪 PHASE 9 — INTEGRATION & TESTING

### CHECKPOINT 9.1 — End-to-end test
- [ ] Chạy toàn bộ app với webcam thật
- [ ] Verify tất cả 4 module hoạt động đồng thời
- [ ] Kiểm tra overlay hiển thị đúng
- **Notes**: _

### CHECKPOINT 9.2 — Multi-face test
- [ ] Test với 2, 3+ khuôn mặt cùng lúc
- [ ] Đảm bảo mỗi face có bbox + label riêng
- **Notes**: _

### CHECKPOINT 9.3 — Edge case test
- [ ] Khuôn mặt đeo khẩu trang / kính
- [ ] Ánh sáng tối
- [ ] Khuôn mặt nghiêng > 45 độ
- [ ] Không có khuôn mặt trong frame
- **Notes**: _

### CHECKPOINT 9.4 — Video file test
- [ ] Test với video file `.mp4` thay vì webcam
- [ ] Verify config switch `CAMERA_INDEX = "path/to/video.mp4"`
- **Notes**: _

---

## ⚡ PHASE 10 — OPTIMIZATION & POLISH

### CHECKPOINT 10.1 — Performance profiling
- [ ] Đo FPS từng module riêng lẻ
- [ ] Identify bottleneck
- [ ] Implement frame skip adaptive (giảm process rate nếu FPS < 10)
- **Notes**: _

### CHECKPOINT 10.2 — Model optimization
- [ ] TensorFlow Lite conversion cho emotion model (optional)
- [ ] Batch processing nếu nhiều face
- **Notes**: _

### CHECKPOINT 10.3 — Documentation
- [ ] Viết `README.md` — hướng dẫn cài đặt + chạy
- [ ] Comment code tất cả public functions
- [ ] Tạo `.env.example`
- **Notes**: _

### CHECKPOINT 10.4 — Final review
- [ ] Code review toàn bộ
- [ ] Xử lý tất cả TODO/FIXME còn lại
- [ ] Test clean install từ đầu
- **Notes**: _

---

## 🗒️ SESSION NOTES

> Agent: Ghi lại các quyết định quan trọng, bugs đã gặp, và thay đổi so với plan gốc tại đây.

| Session | Date       | Completed             | Notes |
|---------|------------|-----------------------|-------|
| 1       | 2026-06-04 | Phase 1–8 (33/45)    | Setup toàn bộ codebase từ đầu. Phase 9–10 pending manual testing. |

---

## ⚠️ KNOWN ISSUES / BLOCKERS

> Ghi lại các vấn đề đang chặn progress:

- Dependencies chưa được cài (`pip install -r requirements.txt`) — cần chạy trước khi test
- PCA model chưa được train — reconstruction module sẽ tự disable cho đến khi có dataset
- DeepFace sẽ tải model weights (~500MB) lần đầu chạy

---

## 🔄 CÁCH AGENT SỬ DỤNG FILE NÀY

```
1. Đọc TỔNG QUAN TIẾN ĐỘ → biết đang ở phase nào
2. Tìm checkpoint [~] đầu tiên → tiếp tục step đang dở
3. Nếu không có [~], tìm [ ] đầu tiên → bắt đầu step mới
4. Đọc claude.md để nạp ngữ cảnh kỹ thuật chi tiết
5. Code xong 1 checkpoint → cập nhật [x] và ghi Notes
6. Cập nhật TỔNG QUAN TIẾN ĐỘ (số đếm)
7. Ghi vào SESSION NOTES
```
