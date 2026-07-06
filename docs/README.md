# 🎓 KỊCH BẢN THUYẾT TRÌNH ĐỒ ÁN CUỐI KỲ (15-20 Phút)
**Môn học:** Thị giác máy tính (Computer Vision)
**Chủ đề:** HOG + SVM (Phát hiện và Phân loại đối tượng)
**Đề tài mở rộng:** Điểm danh khuôn mặt & Chống giả mạo (Anti-Spoofing)
**Nhóm thực hiện:** Khải (Backend/Streamlit) & Đức (Data/SVM)

---

## 🚀 1. Lệnh Chạy Demo (Lưu sẵn để mở ngay lúc báo cáo)
**Khải** chuẩn bị sẵn Terminal và gõ lệnh này (khoảng 30 giây trước khi bắt đầu demo để hệ thống load model):
```bash
# Đảm bảo đã kích hoạt môi trường ảo (venv)
streamlit run app.py
```
> *Mẹo:* Lúc thuyết trình, hãy mở sẵn tab **"Camera giám sát"** để thầy cô thấy được tính năng chạy thời gian thực (Real-time).

---

## ⏱️ 2. PHÂN BỔ THỜI GIAN THUYẾT TRÌNH (15 PHÚT)

### Phần 1: Đặt vấn đề & Giới thiệu (2-3 phút) - *[Đức hoặc Khải nói]*
*   **Chào hỏi:** Chào thầy và các bạn, nhóm chúng em gồm Khải và Đức xin trình bày Chủ đề 2: Dùng HOG và SVM để phát hiện đối tượng.
*   **Lý do chọn đề tài:** Thay vì làm phát hiện người đi bộ (Pedestrian) khá cơ bản, nhóm em quyết định mở rộng làm bài toán **Hệ thống Điểm danh Khuôn mặt**. 
*   **Điểm nhấn:** Đặc biệt, nhóm không chỉ nhận diện danh tính (Identity) mà còn giải quyết một bài toán rất thực tế là **Anti-spoofing (Chống giả mạo)**: phát hiện kẻ gian dùng ảnh in giấy hoặc màn hình điện thoại để qua mặt camera.

### Phần 2: Giải thích Luồng thuật toán Pipeline (5-7 phút)
**Khải trình bày (Phần Xử lý ảnh - Computer Vision):**
1.  **Quét khuôn mặt (Haar Cascade):** Khi ảnh/video đưa vào, hệ thống chuyển sang ảnh xám và dùng thuật toán Haar Cascade quét các đặc trưng sáng/tối (mắt, mũi). Khi phát hiện mặt, nó sẽ cắt (crop) và **resize chuẩn về 64x64 pixel**.
2.  **Rút trích đặc trưng (HOG):** Ảnh 64x64 không đem so sánh pixel trực tiếp, mà được đưa qua HOG. HOG sẽ quét độ dốc (gradient) các góc cạnh khuôn mặt, đếm phân bố của chúng và nén lại thành một vector duy nhất dài **1764 chiều**. Bất cứ khuôn mặt nào đưa vào cũng sẽ bị mã hóa thành 1764 con số này.

**Đức trình bày (Phần Học máy - Machine Learning):**
3.  **So sánh và Phân loại bằng AI (SVM):** Vector 1764 chiều này được ném vào 2 mô hình SVM độc lập mà nhóm đã train sẵn:
    *   **SVM 1 (Định danh - Multi-class):** Phân loại vector này thuộc về Khải, Đức hay là Unknown (Người lạ).
    *   **SVM 2 (Chống giả mạo - Binary):** Kiểm tra xem phân bố HOG này là bề mặt da thật (Real), hay chứa các vân nhiễu Moiré/tia lóa đặc trưng của màn hình điện thoại (Spoof).

### Phần 3: Chạy Demo trực tiếp (3-5 phút) - *[Khải thao tác máy]*
*   **Bước 1:** Khải bật tab Camera, đưa mặt mình vào -> Báo Xanh lá + Tên Khải. (Database bên cạnh sẽ load ảnh mẫu của Khải).
*   **Bước 2:** Đức đưa mặt vào -> Báo Xanh lá + Tên Đức.
*   **Bước 3 (Ăn điểm):** Khải lấy điện thoại, bật 1 bức ảnh của Khải/Đức lên và giơ trước Camera -> Hệ thống phát hiện đây là khuôn mặt Khải/Đức, nhưng vẽ **Khung màu Cam (Spoof)** cảnh báo đây là giả mạo!
*   **Bước 4:** Đức nhờ 1 bạn khác trong lớp (không có trong data) ló mặt vào -> Hệ thống báo **Khung màu Đỏ (KHONG HOP LE)**.

---

## 🛡️ 3. CHUẨN BỊ VẤN ĐÁP (5 PHÚT Q&A - BẢO VỆ ĐỒ ÁN)

### 🔴 Câu hỏi cho Khải (Về Ảnh / Web / Pipeline)
**Q1: Tại sao phải Resize ảnh về 64x64 trước khi làm HOG?**
> **Khải đáp:** Dạ thưa thầy, thuật toán học máy SVM yêu cầu đầu vào phải có số chiều cố định. Đặc trưng HOG phụ thuộc vào kích thước ảnh (ảnh to sẽ sinh ra mảng số to). Do đó, nhóm bắt buộc phải quy hoạch tất cả khuôn mặt bị cắt ra về chung kích thước 64x64. Kích thước này đủ nhỏ để tính toán nhanh (real-time) và đủ lớn để giữ lại các đặc trưng cấu trúc mặt.

**Q2: Số 1764 chiều của HOG từ đâu ra? Em tính như nào?**
> **Khải đáp:** Ảnh 64x64 được chia thành các ô nhỏ (cell) 8x8. Các cell được gom thành block 16x16 (chứa 4 cells). Khi trượt block dọc theo ảnh với bước nhảy (stride) = 8, em sẽ có tổng cộng 7x7 = 49 blocks. Mỗi cell tính ra 9 giá trị góc (9 bins), nên 1 block = 4x9 = 36 giá trị. Tổng cộng toàn bộ ảnh là: 49 blocks × 36 = 1764 chiều.

**Q3: Giao diện Web em dùng framework gì? Tại sao luồng Camera không bị giật lag?**
> **Khải đáp:** Nhóm em dùng Streamlit của Python vì nó hỗ trợ nhúng thẳng xử lý mảng (Numpy/OpenCV) lên web mà không cần viết API backend phức tạp. Để không bị lag, em đã tối ưu hạ độ phân giải luồng camera OpenCV xuống 640x480 để giảm tải xử lý, đảm bảo HOG quét mượt mà.

---

### 🔵 Câu hỏi cho Đức (Về Data / Mô hình ML)
**Q4: Tại sao lại dùng HOG để phát hiện giả mạo (Anti-spoofing) mà không phải màu sắc?**
> **Đức đáp:** Dạ, nếu dùng màu sắc rất dễ bị lừa nếu in ảnh màu. HOG rất mạnh trong việc nắm bắt kết cấu bề mặt (Texture). Màn hình thiết bị điện tử luôn sinh ra hiện tượng nhiễu vân Moiré (các sọc mờ) hoặc lóa sáng phản chiếu mà mắt thường khó để ý, nhưng độ dốc vi phân (Gradient) của HOG sẽ bắt được các góc bất thường này. Da người thật thì có độ phản xạ ánh sáng (Lambertian) khuếch tán đều đặn hơn. 

**Q5: Tại sao nhóm phải dùng đến 2 mô hình SVM riêng biệt? Dùng 1 cái có được không?**
> **Đức đáp:** Dạ được nhưng sẽ kém hiệu quả. Nhận diện danh tính là bài toán phân loại đa lớp (Multi-class), HOG cần tập trung vào đường nét mắt/mũi. Trong khi đó, chống giả mạo là bài toán phân loại nhị phân (Binary - Thật/Giả), HOG cần tập trung vào nhiễu bề mặt. Việc dùng 2 kernel tuyến tính (`LinearSVC`) chạy song song vừa nhanh nhẹn, vừa giúp mỗi model tập trung tối ưu cực đại cho 1 nhiệm vụ chuyên biệt.

**Q6: Làm sao để xử lý tình huống số lượng ảnh mặt thật nhiều hơn hẳn ảnh mặt giả mạo (Class Imbalance)?**
> **Đức đáp:** Quá trình huấn luyện em dùng tham số `class_weight='balanced'` trong thư viện scikit-learn. Nó sẽ tự động tính toán và phạt nặng hơn nếu mô hình đoán sai lớp thiểu số (ảnh giả), ép siêu phẳng (hyperplane) của SVM phải phân chia công bằng cho cả 2 loại.

---
*Chúc hai bạn tự tin thuyết trình và bảo vệ đồ án thành công!*
