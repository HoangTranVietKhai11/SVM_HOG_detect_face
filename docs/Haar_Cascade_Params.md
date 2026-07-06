# 🧠 TÀI LIỆU VẤN ĐÁP MỞ RỘNG: HIỆU CHỈNH HAAR CASCADE
**Dành cho:** Khải & Đức (Sử dụng khi giảng viên hỏi về cách tuỳ biến độ nhạy của Camera)

---

## 📂 1. Tùy chỉnh thông số ở đâu?
Trong dự án của chúng ta, thuật toán quét khuôn mặt (`detectMultiScale`) được gọi ở **2 nơi khác nhau** phục vụ cho 2 mục đích. Bạn có thể mở các file này lên để tự tay thay đổi các con số:

1. **[face_detector.py](file:///d:/SVM_HOG_detect_face/face_detector.py#L30-L37)** (Dòng ~33): Nơi xử lý tập dữ liệu (dataset) để train mô hình.
2. **[app.py](file:///d:/SVM_HOG_detect_face/app.py#L105-L109)** (Dòng ~109): Nơi xử lý Camera thời gian thực (Real-time) trên web.

> [!TIP]
> Việc cài đặt 2 mức độ nhạy khác nhau cho Dataset và Web App thể hiện tính tinh tế của người làm dự án. Dataset cần khắt khe để dữ liệu sạch, còn Web App cần linh hoạt để nhận diện nhanh.

---

## ⚙️ 2. Bản chất các tham số (Đọc để hiểu và chém gió)

Khi hàm `cv2.CascadeClassifier.detectMultiScale()` chạy, nó dựa vào 3 tham số sống còn sau:

### Tham số 1: `scaleFactor` (Tốc độ thu nhỏ ảnh)
*   **Bản chất:** Hệ thống sẽ tạo ra một "Kim tự tháp ảnh" (Image Pyramid) bằng cách thu nhỏ ảnh gốc đi nhiều lần để quét được cả khuôn mặt to (đứng gần) và khuôn mặt bé (đứng xa).
*   **Cách chỉnh:** 
    *   `1.1` (Tức là giảm 10% mỗi vòng lặp): Quét nhanh hơn, tiết kiệm CPU nhưng dễ bỏ lỡ khuôn mặt.
    *   `1.05` (Giảm 5% mỗi vòng): Quét rất dày đặc và chi tiết. Máy tính phải tính toán cực nhiều nhưng bù lại bắt dính được khuôn mặt đang chuyển động.

### Tham số 2: `minNeighbors` (Bộ lọc nhiễu trùng lặp)
*   **Bản chất:** Thuật toán Haar Cascade rất "nhạy". Khi quét qua một khuôn mặt thật, nó thường sinh ra hàng chục cái ô vuông đè lên nhau. Tham số này quy định: *Phải có bao nhiêu ô vuông xếp chồng lên nhau thì mới công nhận đó là khuôn mặt thật?*
*   **Cách chỉnh:**
    *   `5` (Khắt khe): Giúp không bị nhận nhầm cái quạt hay đồng hồ thành khuôn mặt. Nhưng lỡ bạn đeo kính hoặc quay mặt hơi nghiêng thì thuật toán chê và bỏ qua luôn.
    *   `3` (Nhạy bén): Rất dễ dãi, phù hợp cho Webcam độ phân giải thấp hoặc chụp selfie cự ly gần có đeo kính.

### Tham số 3: `minSize` & `maxSize` (Giới hạn khung bắt)
*   **Bản chất:** Lọc bỏ rác.
*   **Cách chỉnh:** Cài `minSize=(30, 30)` để nó không thèm quét những vật thể nhỏ hơn 30x30 pixel (thường là nhiễu hạt sau background).

---

## 🎤 3. Văn mẫu trả lời Vấn Đáp
Nếu thầy giáo hỏi: *"Tại sao lúc đưa ảnh selfie vào thì không nhận diện được, em đã fix lỗi đó bằng cách nào?"*

**Bạn trả lời:**
> *"Dạ thưa thầy, mặc định thư viện OpenCV khuyên dùng `scaleFactor=1.1` và `minNeighbors=5`. Ban đầu nhóm em áp dụng mức khắt khe này để quét Dataset nhằm loại bỏ sạch sẽ ảnh rác. 
>
> Tuy nhiên, khi đưa vào thực tế chạy Camera Streamlit hoặc chụp ảnh Selfie có mang kính, các đặc trưng mắt/mũi bị che khuất dẫn đến số lượng khung viền (neighbors) bắt được bị giảm xuống dưới 5. Kết quả là thuật toán tưởng nhầm đó không phải là mặt. 
> 
> Để khắc phục, em đã chủ động hạ `minNeighbors` xuống `3` và giảm `scaleFactor` về `1.05` để thuật toán quét chậm lại, kỹ hơn và dễ dàng chấp nhận các khuôn mặt bị che khuất một phần ạ."*
