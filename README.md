# 🎭 Ứng dụng Điểm danh Khuôn mặt tích hợp Anti-Spoofing

> Đồ án cuối kỳ | HOG + SVM + Streamlit Web App
> Nhóm thực hiện: **Hoàng Trần Việt Khải** & **Vũ Huy Đức**

Web app cho phép upload ảnh khuôn mặt → phát hiện khuôn mặt → nhận diện danh tính (Khải / Đức / Unknown) → phán đoán Thật / Giả mạo → hiển thị bounding box + nhãn kết quả trực quan.

⚠️ **Định hướng dự án:** Tập trung toàn lực vào HOG + SVM chạy đúng. Thầy ấn tượng thuật toán hơn giao diện đẹp. Không dùng Flutter, không dùng FastAPI — tất cả gộp trong 1 app Python.

---

## 🏗️ Kiến trúc

- **Frontend:** Streamlit (Python) — upload ảnh, hiển thị kết quả
- **Backend:** OpenCV + scikit-learn — HOG extraction + SVM predict
- **Face Detection:** Haar Cascade (có sẵn trong OpenCV)
- **Classification:** 2 mô hình SVM riêng biệt (nhận diện danh tính + liveness detection)

---

## 📁 Cấu trúc thư mục dự án

```
project/
├── dataset/
│   ├── real/
│   │   ├── khai/          ← 50 ảnh khuôn mặt thật của Khải
│   │   └── duc/            ← 50 ảnh khuôn mặt thật của Đức
│   └── spoof/               ← 50 ảnh giả mạo qua màn hình
├── models/
│   ├── model_identity.pkl   ← SVM nhận diện danh tính
│   └── model_liveness.pkl   ← SVM phân loại thật/giả
├── face_detector.py          ← Haar Cascade detect khuôn mặt        [Khải]
├── hog_extractor.py           ← Trích xuất HOG vector                [Khải]
├── preprocess.py               ← Tiền xử lý + trích HOG dataset        [Đức]
├── train_svm.py                 ← Huấn luyện 2 model SVM                [Đức]
├── predict.py                    ← Load model + chạy pipeline             [Khải]
├── app.py                         ← Streamlit web app                      [Khải]
└── README.md
```

---

## ⚙️ Cài đặt môi trường

```bash
pip install opencv-python scikit-learn streamlit numpy joblib
```

## 🚀 Chạy ứng dụng

```bash
streamlit run app.py
```

---

## 👥 Phân công nhiệm vụ & Timeline

### 🟦 Hoàng Trần Việt Khải — Backend + Web App

| Ngày | Công việc |
|---|---|
| **Thứ 6 (hôm nay)** | Cài môi trường + viết `face_detector.py` (Haar Cascade): detect vùng khuôn mặt → cắt ROI → resize 64×64. Test trên 3–4 ảnh. |
| **Thứ 7** | Viết `hog_extractor.py` (HOG params: `winSize=(64,64)`, `blockSize=(16,16)`, `cellSize=(8,8)`, `nbins=9`) và `predict.py` (load 2 model `.pkl` từ Đức → pipeline detect → HOG → predict). |
| **Chủ nhật** | Viết `app.py` bằng Streamlit: upload ảnh, gọi `predict.py`, vẽ bounding box xanh (Thật) / đỏ (Giả) + nhãn danh tính. Test end-to-end. |
| **Thứ 2 – Thứ 3** | Fix bug, chụp ảnh demo, viết báo cáo phần Face Detection + HOG + kiến trúc web app. Ôn vấn đáp. |

⚠️ **Lưu ý kỹ thuật:** Hiểu rõ tại sao resize về 64×64 — HOG descriptor cần kích thước cố định để vector đặc trưng có cùng chiều dài cho tất cả ảnh. Đây là phần lý thuyết cốt lõi, cần hiểu từng tham số HOG để trả lời vấn đáp.

### 🟩 Vũ Huy Đức — Data + Train SVM

| Ngày | Công việc |
|---|---|
| **Thứ 6 (hôm nay)** | Thu thập dataset: 50 ảnh thật của Khải, 50 ảnh thật của Đức (đa dạng góc độ/ánh sáng/biểu cảm), 50 ảnh giả mạo (chụp lại ảnh phát trên điện thoại/laptop). Tổ chức vào `dataset/real/khai/`, `dataset/real/duc/`, `dataset/spoof/`. |
| **Thứ 7** | Viết `preprocess.py` (Haar Cascade cắt mặt → resize 64×64 → trích HOG → lưu `X_train.npy`, `y_train.npy`). Train **SVM Model 1** (nhận diện danh tính: khai/duc/unknown) dùng `sklearn.svm.LinearSVC`, đánh giá accuracy + confusion matrix, lưu `model_identity.pkl`. |
| **Chủ nhật** | Train **SVM Model 2** (liveness detection: real/spoof), đánh giá accuracy/F1/confusion matrix, lưu `model_liveness.pkl`. **Gửi 2 file `.pkl` cho Khải trước tối Chủ nhật** để kịp test pipeline. |
| **Thứ 2 – Thứ 3** | Viết báo cáo phần lý thuyết SVM + quá trình thu thập data + bảng kết quả accuracy/F1. Ôn vấn đáp. Hỗ trợ Khải nếu lỗi liên quan định dạng HOG vector. |

⚠️ **Deadline quan trọng:** Giao file `.pkl` cho Khải chậm nhất tối Chủ nhật, để Khải có thời gian test pipeline.

---

## 🎯 Thứ tự ưu tiên nếu không kịp tiến độ

Nếu bị trễ, cắt theo thứ tự — phần trên quan trọng hơn phần dưới:

| TT | Hạng mục | Mức độ | Lý do |
|---|---|---|---|
| 1 | HOG vector trích xuất đúng | Bắt buộc | 30% điểm lý thuyết |
| 2 | SVM train được, accuracy hợp lý | Bắt buộc | 40% điểm thuật toán |
| 3 | Web app Streamlit chạy được | Quan trọng | Demo thực tế |
| 4 | Liveness Detection (Anti-Spoofing) | Tùy thời gian | 20% sáng tạo |
| 5 | Giao diện đẹp | Không cần thiết | Không ảnh hưởng điểm |

---

## 🎓 Chuẩn bị vấn đáp

### Hoàng Trần Việt Khải cần trả lời được

- **Haar Cascade:** bộ phát hiện đặc trưng được train trước trên hàng nghìn ảnh khuôn mặt, nhận diện dựa trên các đặc trưng Haar (sự chênh lệch độ sáng giữa các vùng ảnh liền kề).
- **Tại sao resize về 64×64:** HOG descriptor yêu cầu kích thước đầu vào cố định để tạo ra vector đặc trưng có cùng chiều dài — SVM chỉ nhận input có số chiều nhất quán.
- **blockSize=(16,16), cellSize=(8,8) nghĩa là gì:** mỗi cell 8×8 pixel, mỗi block gồm 2×2 cells = 16×16. Với window 64×64 sẽ có 7×7 = 49 blocks, mỗi block 4 cells × 9 bins = 36 chiều, tổng vector = 49×36 = 1764 chiều.
- **Streamlit:** thư viện Python tạo web app nhanh, cho phép nhúng code xử lý ảnh trực tiếp mà không cần viết HTML/CSS hay REST API riêng.

### Vũ Huy Đức cần trả lời được

- **C parameter trong SVM:** kiểm soát trade-off giữa margin rộng và lỗi phân loại. C nhỏ → margin rộng, chấp nhận nhiều lỗi hơn (underfit). C lớn → margin hẹp, cố gắng phân loại đúng hết (dễ overfit).
- **Tại sao 2 SVM riêng biệt:** vì 2 bài toán khác nhau về bản chất — nhận diện danh tính là multi-class (phân 3+ lớp), liveness detection là binary (thật/giả). Gộp lại thì vector đặc trưng cần học 2 pattern khác nhau cùng lúc, kém hiệu quả hơn.
- **Class imbalance:** nếu ảnh thật nhiều hơn ảnh giả nhiều, model dễ bias về class đa số. Xử lý bằng `class_weight='balanced'` trong `LinearSVC` hoặc cân bằng số lượng ảnh khi thu thập.
- **Tại sao HOG phát hiện giả mạo được:** màn hình điện thoại/laptop tạo ra vân Moiré và phản xạ đặc trưng khác hoàn toàn với cấu trúc bề mặt da người — HOG bắt được sự khác biệt này qua phân bố hướng gradient.

---

## 📌 Ghi chú

- Deadline báo cáo: **Thứ 4 tuần sau**.
- Mọi thay đổi về pipeline (format HOG vector, tên file model...) cần thông báo ngay cho cả nhóm để tránh lỗi tích hợp.
