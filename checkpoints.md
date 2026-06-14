# ✅ CHECKPOINTS — FaceVision AI Project

> **Hướng dẫn cho Agent**: Đọc file này để biết đã hoàn thành đến đâu.
> - `[ ]` = Chưa làm
> - `[~]` = Đang làm / Làm dở
> - `[x]` = Đã hoàn thành
>
> **Khi bắt đầu session mới**:
> 1. Đọc `claude.md` để nạp ngữ cảnh kỹ thuật
> 2. Tìm checkpoint `[~]` gần nhất → tiếp tục
> 3. Nếu không có `[~]`, tìm `[ ]` đầu tiên → bắt đầu
> 4. Sau khi xong 1 checkpoint: cập nhật `[x]`, ghi Notes, cập nhật TỔNG QUAN

---

## 📊 TỔNG QUAN TIẾN ĐỘ

```
Phase 1:  Setup & Structure       [x] 5/5  ✅ Hoàn thành
Phase 2:  Core Pipeline           [x] 3/4  ⚠️  test pending
Phase 3:  Module Detection        [x] 3/4  ⚠️  test pending
Phase 4:  Module Recognition      [x] 4/5  ⚠️  test pending
Phase 5:  Module Emotion          [x] 3/4  ⚠️  test pending
Phase 6:  Module Reconstruction   [x] 4/5  ⚠️  test pending
Phase 7:  Flask API & Stream      [x] 3/5  ⚠️  cần chạy app
Phase 8:  Frontend UI             [x] 5/5  ✅ Hoàn thành
Phase 9:  Integration & Test      [ ] 0/4  ← LÀM TIẾP THEO (cần chạy app)
Phase 10: Optimization            [ ] 0/4
Phase 11: Thực nghiệm E1–E4       [~] 2/8  ← ĐANG LÀM
Phase 12: Kết quả vào báo cáo     [ ] 0/4

TỔNG: 35/53 checkpoints (66%)
```

> ⚠️ Phase 11–12 là mới — thêm vào để phục vụ báo cáo học thuật.

---

## 🚀 PHASE 1 — PROJECT SETUP & STRUCTURE ✅

### CHECKPOINT 1.1 [x] — Khởi tạo cấu trúc thư mục
- **Notes**: Tạo core/, modules/, models/{known_faces,embeddings,eigenfaces}, static/{css,js,assets}, templates/, utils/, tests/, scripts/

### CHECKPOINT 1.2 [x] — File cấu hình
- **Notes**: config.py đủ section: video, detection, recognition, emotion, reconstruction, flask, paths, emotion_colors

### CHECKPOINT 1.3 [x] — Utility modules
- **Notes**: logger.py dùng ANSI colors; image_utils.py có resize_frame, normalize_face, crop_face, align_face, encode_jpeg

### CHECKPOINT 1.4 [x] — Database khuôn mặt
- **Notes**: FaceDatabase class với load(), reload(), save(), find_closest() cosine distance; dataclass FaceRecord, FaceRegion, EmotionResult

### CHECKPOINT 1.5 [x/~] — Kiểm tra môi trường
- [x] Tạo `tests/test_env.py`
- [ ] Chạy `python tests/test_env.py` trên máy GPU mới và xác nhận PASS
- **Notes**: Cần verify lại trên máy mới. Chạy: `python tests/test_env.py`

---

## 🔧 PHASE 2 — CORE PIPELINE ✅ (code xong, test pending)

### CHECKPOINT 2.1 [x] — Camera module
- **Notes**: CameraStream.start(), read(), stop(). Frame queue maxsize=2. Support webcam index và video path.

### CHECKPOINT 2.2 [x] — Overlay renderer
- **Notes**: draw_face_box(), draw_fps(), draw_face_count(), draw_reconstruction_thumbnail()

### CHECKPOINT 2.3 [x] — Pipeline orchestrator
- **Notes**: FacePipeline — cache kết quả giữa các frame, lazy module loading, error isolation per module

### CHECKPOINT 2.4 [ ] — Pipeline unit test
- [ ] Test `process_frame` với ảnh tĩnh
- [ ] Xác nhận output format đúng
- **Notes**: Chạy sau khi cài dependencies: `python -c "from core.pipeline import FacePipeline; p = FacePipeline(); print('OK')"`

---

## 👁️ PHASE 3 — MODULE DETECTION ✅ (code xong, test pending)

### CHECKPOINT 3.1 [x] — Haar Cascade
- **Notes**: FaceDetector._detect_haar() với haarcascade_frontalface_default.xml

### CHECKPOINT 3.2 [x] — MTCNN
- **Notes**: FaceDetector._detect_mtcnn() — resize 50%, detect, scale bbox x2, fallback sang Haar nếu fail

### CHECKPOINT 3.3 [x] — Face crop + align
- **Notes**: crop_face() và align_face() trong utils/image_utils.py

### CHECKPOINT 3.4 [ ] — Detection test
- [ ] Chạy `python tests/test_detection.py`
- [ ] Benchmark FPS: Haar vs MTCNN trên cùng ảnh test
- **Notes**: Cần dataset ảnh. Dùng ảnh trong experiments/dataset/ nếu đã có.

---

## 🪪 PHASE 4 — MODULE RECOGNITION ✅ (code xong, test pending)

### CHECKPOINT 4.1 [x] — ArcFace embedding
- **Notes**: extract_embedding() dùng temp file + DeepFace.represent, detector_backend="skip"

### CHECKPOINT 4.2 [x] — Face registration
- **Notes**: register_face() lưu {name}_{timestamp}.npy; scripts/register_face.py CLI

### CHECKPOINT 4.3 [x] — Face identification
- **Notes**: identify_face() → (name, cosine_distance); "Unknown" nếu distance > 0.40

### CHECKPOINT 4.4 [x] — Embedding cache
- **Notes**: FaceDatabase trong utils/face_db.py — load 1 lần, reload() khi thêm face mới

### CHECKPOINT 4.5 [ ] — Recognition test
- [ ] Đăng ký 2–3 người bằng `python scripts/register_face.py`
- [ ] Chạy `python tests/test_recognition.py`
- [ ] Test edge case: cùng người ảnh khác / người lạ
- **Notes**: Cần dataset ảnh trước.

---

## 😊 PHASE 5 — MODULE EMOTION ✅ (code xong, test pending)

### CHECKPOINT 5.1 [x] — DeepFace emotion analyzer
- **Notes**: DeepFace.analyze actions=["emotion"], detector_backend="skip"

### CHECKPOINT 5.2 [x] — Smoothing
- **Notes**: EmotionTracker dùng deque(maxlen=5), rolling average

### CHECKPOINT 5.3 [x] — Color mapping
- **Notes**: EMOTION_COLORS dict trong config.py, dùng trong overlay.py

### CHECKPOINT 5.4 [ ] — Emotion test
- [ ] Chạy `python tests/test_emotion.py` với ảnh biểu cảm rõ ràng
- **Notes**: Dùng ảnh sample từ internet (vd: FER2013 sample images) nếu chưa có dataset.

---

## 🧬 PHASE 6 — MODULE RECONSTRUCTION (code xong, chưa train)

### CHECKPOINT 6.1 [x/~] — PCA training
- [x] Implement train() method
- [ ] Tạo `scripts/train_eigenfaces.py` (script CLI để train từ dataset)
- [ ] Train PCA model và lưu `models/eigenfaces/pca_model.pkl`
- **Notes**: Module tự disable nếu pkl chưa tồn tại. Cần dataset trước.

### CHECKPOINT 6.2 [x] — Face reconstruction
- **Notes**: reconstruct() returns grayscale 64×64 numpy array

### CHECKPOINT 6.3 [x/~] — Reconstruction error
- [x] get_reconstruction_error() → MSE float
- [ ] Xác định threshold để flag "unusual face" (dựa trên phân phối MSE trên dataset)
- **Notes**: Chọn threshold = mean + 2*std của MSE trên validation set

### CHECKPOINT 6.4 [x/~] — Visualization
- [x] draw_reconstruction_thumbnail() trong overlay.py
- [ ] Eigenface components visualization (optional, cho báo cáo)

### CHECKPOINT 6.5 [ ] — Reconstruction test
- [ ] Train PCA model với dataset ảnh
- [ ] Test reconstruct() → so sánh ảnh gốc vs tái dựng bằng mắt
- **Notes**: Cần hoàn thành Phase 11 E4 trước.

---

## 🌐 PHASE 7 — FLASK API & VIDEO STREAM

### CHECKPOINT 7.1 [x] — Flask app skeleton
- **Notes**: atexit.register(on_exit) để release camera. FacePipeline load lúc startup.

### CHECKPOINT 7.2 [x] — MJPEG stream
- **Notes**: generate_frames() generator, Content-Type: multipart/x-mixed-replace

### CHECKPOINT 7.3 [x] — API endpoints
- **Notes**: /api/status, /api/faces, /api/register (POST), /api/snapshot

### CHECKPOINT 7.4 [ ] — API test
- [ ] Chạy `python app.py`
- [ ] Test: `curl http://localhost:5000/api/status`
- [ ] Test: `curl http://localhost:5000/api/faces`
- [ ] Test upload: `curl -X POST -F "name=test" -F "image=@test.jpg" http://localhost:5000/api/register`
- **Notes**: Cần chạy trên máy có webcam.

### CHECKPOINT 7.5 [ ] — Stream stability
- [ ] Stream liên tục 5 phút không memory leak
- [ ] FPS ổn định > 15 FPS (GPU: > 25 FPS kỳ vọng)
- **Notes**: Dùng `htop` hoặc Task Manager để monitor RAM trong lúc stream.

---

## 🎨 PHASE 8 — FRONTEND UI ✅

- **Notes**: index.html split layout 70/30; style.css dark theme #0d0f14 accent #00e5cc; main.js polling + upload + toast. Hoàn chỉnh.

---

## 🧪 PHASE 9 — INTEGRATION & TESTING ← LÀM TIẾP THEO

### CHECKPOINT 9.0 [ ] — Cài đặt máy GPU mới (làm đầu tiên)
- [ ] `git clone https://github.com/letuyenvuong/facevision-ai.git`
- [ ] `python -m venv venv && source venv/bin/activate`
- [ ] `pip install -r requirements.txt`
- [ ] `pip install matplotlib seaborn pandas` (thêm nếu chưa có)
- [ ] `python tests/test_env.py` → xác nhận PASS
- **Notes**: DeepFace download ~500MB lần đầu. Cần internet.

### CHECKPOINT 9.1 [ ] — End-to-end test webcam
- [ ] `python app.py` → mở http://localhost:5000
- [ ] Verify webcam stream hiển thị trong browser
- [ ] Verify detection box xuất hiện khi có mặt người
- [ ] Verify emotion label thay đổi theo biểu cảm
- [ ] Verify FPS counter hoạt động (GPU kỳ vọng > 25 FPS)
- **Notes**: Chụp screenshot kết quả để ghi vào báo cáo.

### CHECKPOINT 9.2 [ ] — Multi-face test
- [ ] Test với 2, 3+ khuôn mặt cùng lúc
- [ ] Mỗi face có bbox + label riêng biệt
- [ ] Ghi nhận FPS khi có nhiều face (so sánh với 1 face)
- **Notes**: Dùng ảnh nhóm in ra giấy nếu không đủ người.

### CHECKPOINT 9.3 [ ] — Edge case test
- [ ] Khẩu trang: detection có bỏ sót không?
- [ ] Kính: emotion có bị ảnh hưởng không?
- [ ] Ánh sáng tối: thử che bớt đèn phòng
- [ ] Không có mặt: pipeline xử lý đúng (không crash)
- [ ] Mặt nghiêng > 45°: MTCNN có fallback sang Haar không?
- **Notes**: Ghi lại quan sát cho §7.3 kịch bản A–F trong báo cáo.

### CHECKPOINT 9.4 [ ] — Video file test
- [ ] Test `CAMERA_INDEX = "path/to/test.mp4"` trong config.py
- [ ] Verify stream từ video file chạy đúng
- **Notes**: Tải 1 video test ngắn từ internet nếu không có sẵn.

---

## ⚡ PHASE 10 — OPTIMIZATION

### CHECKPOINT 10.1 [ ] — Performance profiling
- [ ] Đo thời gian từng module: detection, embedding, emotion, reconstruction (ms/frame)
- [ ] Ghi vào bảng: Module | Avg ms | % tổng pipeline
- [ ] Identify bottleneck (thường là DeepFace embedding)
- **Notes**: Thêm `time.perf_counter()` wrap quanh từng module trong pipeline.py

### CHECKPOINT 10.2 [ ] — GPU optimization
- [ ] Verify TensorFlow dùng GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- [ ] Nếu GPU available: thử tăng batch size hoặc giảm PROCESS_EVERY_N_FRAMES
- [ ] So sánh FPS CPU-only vs GPU
- **Notes**: Ghi FPS GPU vs CPU vào báo cáo §7.2 cấu hình thực nghiệm.

### CHECKPOINT 10.3 [ ] — Documentation
- [ ] Viết `README.md` — cài đặt, chạy, đăng ký face
- [ ] Thêm `.env.example` nếu có biến môi trường
- **Notes**: README tối thiểu: requirements, install steps, run command.

### CHECKPOINT 10.4 [ ] — Final review
- [ ] Kiểm tra tất cả TODO/FIXME còn lại trong code
- [ ] Test clean install từ đầu trên máy mới
- **Notes**: Chạy `grep -r "TODO\|FIXME" . --include="*.py"` để liệt kê.

---

## 📊 PHASE 11 — THỰC NGHIỆM E1–E4 (MỤC TIÊU CHÍNH CHO BÁO CÁO)

> Đây là phase quan trọng nhất hiện tại. Kết quả đây sẽ điền vào Chương 7 báo cáo Word.
> Xem chi tiết đầy đủ trong `claude.md` mục "4 THỰC NGHIỆM CẦN CHẠY".

### CHECKPOINT 11.0 [x] — Tạo cấu trúc thư mục thực nghiệm
```bash
mkdir -p experiments/{dataset,results/E1_detector,results/E2_alignment,results/E3_emotion,results/E4_pca,figures}
echo "person_id,image_file,session,pose,lighting,occlusion,camera,resolution,note" > experiments/dataset/metadata.csv
```

### CHECKPOINT 11.1 [ ] — Thu thập dataset
- [ ] Ít nhất 5–10 người tham gia (lý tưởng 10–15)
- [ ] Mỗi người: 5 điều kiện × 5–10 ảnh = 25–50 ảnh/người
- [ ] 5 điều kiện: frontal, pose_15, pose_30, dark, occluded
- [ ] Điền metadata.csv cho từng ảnh
- [ ] Tổng tối thiểu: ~250 ảnh
- **Notes**: Lưu ảnh vào `experiments/dataset/person_XX/condition_NNN.jpg`

### CHECKPOINT 11.2 [x] — Viết + chạy script E1 (Detector comparison)
- [x] Viết `experiments/run_E1_detector.py`
- [x] Chạy demo (chưa có dataset thực): `E1_recall_by_condition.png` + `E1_latency_boxplot.png` + `E1_results_DEMO.csv`
- [ ] Chạy lại với dataset thực: `python experiments/run_E1_detector.py`
- **Notes**: Script tự detect khi chưa có dataset → sinh demo output. Cần cài mtcnn cho MTCNN detector.

### CHECKPOINT 11.3 [x] — Viết + chạy script E2 (Alignment impact)
- [x] Viết `experiments/run_E2_alignment.py`
- [x] Chạy demo: `E2_histogram.png` + `E2_roc.png` + `E2_summary.json`
- [ ] Chạy lại với dataset + deepface thực: `python experiments/run_E2_alignment.py`
- **Notes**: Cần `pip install deepface tensorflow` + dataset ảnh.

### CHECKPOINT 11.4 [x] — Viết + chạy script E3 (FER evaluation)
- [x] Viết `experiments/run_E3_emotion.py`
- [x] Chạy demo: `E3_confusion_matrix.png` + `E3_per_class_f1.png` + `E3_results.json`
- [ ] Chuẩn bị ảnh biểu cảm có nhãn → `experiments/dataset/emotion/<label>/*.jpg`
- [ ] Chạy lại với dataset thực: `python experiments/run_E3_emotion.py`
- **Notes**: Có thể dùng subset FER2013 (Kaggle). Cần deepface+tensorflow.

### CHECKPOINT 11.5 [x] — Viết + chạy script E4 (PCA reconstruction)
- [x] Viết `experiments/run_E4_pca.py`
- [x] Chạy demo: `E4_metrics_DEMO.csv` + `E4_curve_mse_ssim.png` + `E4_scree_plot.png` + `E4_reconstruction_grid.png`
- [ ] Chạy lại với dataset thực (không cần deepface/tensorflow!): `python experiments/run_E4_pca.py`
- **Notes**: E4 chỉ cần opencv + sklearn + scikit-image — có thể chạy ngay khi có ảnh dataset.

### CHECKPOINT 11.6 [ ] — Profiling pipeline (bổ sung cho báo cáo)
- [ ] Đo thời gian từng stage: detection / embedding / emotion / reconstruction (ms/frame)
- [ ] Lưu: `experiments/results/profiling_results.csv`
- [ ] Tính % thời gian mỗi stage so với tổng pipeline
- **Notes**: Thêm timer trong pipeline.py, chạy 100 frame rồi lấy trung bình

---

## 📝 PHASE 12 — ĐIỀN KẾT QUẢ VÀO BÁO CÁO WORD

### CHECKPOINT 12.1 [ ] — Điền bảng số liệu Chương 7
- [ ] §7.4 bảng kịch bản S01–S04: điền từ E1 results
- [ ] §4.2 bảng Haar vs MTCNN: điền từ E1_results.csv
- [ ] §4.3 embedding analysis: điền từ E2_summary.json
- [ ] §5.5 confusion matrix: chèn E3_confusion_matrix.png

### CHECKPOINT 12.2 [ ] — Chèn biểu đồ
- [ ] §4.2: chèn E1_recall_by_condition.png
- [ ] §4.3: chèn E2_histogram.png + E2_roc.png
- [ ] §5.5: chèn E3_confusion_matrix.png + E3_per_class_f1.png
- [ ] §6.1: chèn E4_reconstruction_grid.png + E4_curve_mse_ssim.png + E4_scree_plot.png

### CHECKPOINT 12.3 [ ] — Viết phân tích kết quả
- [ ] §7.5 thảo luận: trả lời RQ1–RQ5 bằng số liệu cụ thể
- [ ] §7.7 phân tích thất bại: 3–5 trường hợp lỗi điển hình
- [ ] §10 Kết luận: viết lại với số liệu thực ("MTCNN đạt Recall X% ở pose 30°...")

### CHECKPOINT 12.4 [ ] — Bổ sung lý thuyết còn thiếu
- [ ] §2.2: công thức loss multi-task MTCNN, giải thích NMS
- [ ] §2.3: ma trận affine alignment, phân tích tại sao giảm variance
- [ ] §6.1: công thức explained variance ratio, scree plot
- [ ] §3/§4: giải thích tại sao chọn threshold=0.40, k=50 (từ số liệu E2, E4)

---

## 🗒️ SESSION NOTES

| Session | Date       | Completed                       | Notes |
|---------|------------|---------------------------------|-------|
| 1       | 2026-06-04 | Phase 1–8 (33/45)              | Setup toàn bộ codebase trên máy cũ |
| 2       | 2026-06-09 | Cập nhật claude.md + checkpoints | Chuyển sang máy GPU mới. Thêm Phase 11–12 cho thực nghiệm báo cáo |
| 3       | 2026-06-09 | 11.0, 11.2–11.5 (script E1–E4) | Tạo experiments/. Viết 4 script E1–E4. Demo output chạy OK (15 files). deepface/mtcnn/tensorflow chưa cài. |

---

## ⚠️ KNOWN ISSUES / BLOCKERS

1. **Dependencies chưa cài trên máy GPU mới** → Chạy `pip install -r requirements.txt` trước tiên
2. **PCA model chưa train** → reconstruction module tự disable. Train sau khi có dataset (Phase 11.1)
3. **DeepFace download ~500MB lần đầu** → Cần internet khi chạy lần đầu
4. **matplotlib/seaborn/pandas có thể chưa trong requirements.txt** → `pip install matplotlib seaborn pandas` nếu thiếu
5. **Script E1–E4 chưa tồn tại** → Agent cần viết mới trong Phase 11

---

## 🔄 THỨ TỰ LÀM TIẾP THEO (cho Agent)

```
Bước 1: CHECKPOINT 9.0 — Cài đặt môi trường máy mới
Bước 2: CHECKPOINT 1.5 — Chạy test_env.py, xác nhận PASS
Bước 3: CHECKPOINT 9.1 — Chạy app.py lần đầu, xác nhận webcam stream
Bước 4: CHECKPOINT 11.0 — Tạo thư mục experiments/
Bước 5: CHECKPOINT 11.1 — Thu thập dataset
Bước 6: CHECKPOINT 11.2 → 11.5 — Viết và chạy E1, E2, E3, E4
Bước 7: CHECKPOINT 12.1 → 12.4 — Điền kết quả vào báo cáo Word
```
