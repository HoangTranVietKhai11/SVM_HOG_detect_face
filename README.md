# Hệ Thống Điểm Danh Khuôn Mặt Tích Hợp Anti-Spoofing (Liveness Detection)

Đây là đồ án môn học Thị giác máy tính (Computer Vision) ứng dụng thuật toán trích xuất đặc trưng HOG và bộ phân loại SVM để nhận diện khuôn mặt và phát hiện giả mạo (thật/giả).

## Cấu trúc mã nguồn & Dữ liệu
- `data/raw/`: Thư mục chứa dữ liệu ảnh gốc (person_a, person_b, unknown, v.v.).
- `data/processed/`: Thư mục chứa dữ liệu đã được tiền xử lý dưới dạng ma trận `.npy`.
- `models/`: Thư mục lưu trữ các file mô hình đã được huấn luyện (`.pkl`).
- `docs/`: Chứa file báo cáo đồ án.

## Hướng dẫn cài đặt và chạy hệ thống

### Bước 1: Cài đặt thư viện (Requirements)
Hệ thống yêu cầu cài đặt Python 3. Khởi tạo môi trường ảo (nếu cần) và chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy ứng dụng Giao diện Web (Streamlit)
Bạn có thể thực hiện tất cả các bước (thu thập dữ liệu, tiền xử lý, huấn luyện mô hình, và nhận diện) trực tiếp trên giao diện web:
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở trang web (mặc định tại `localhost:8501`). Hệ thống cung cấp các chức năng chính ở thanh menu bên trái (Sidebar):
- **Thu thập dữ liệu (Capture Faces):** Thêm dữ liệu khuôn mặt cho người mới thông qua webcam.
- **Tiền xử lý (Preprocess):** Cắt mặt, tăng cường dữ liệu và trích xuất đặc trưng HOG.
- **Huấn luyện mô hình (Train):** Huấn luyện mô hình SVM (Identity và Liveness).
- **Trang chủ (Trải nghiệm nhận diện):** Tải ảnh tĩnh hoặc kết nối webcam để điểm danh và nhận diện chống giả mạo thời gian thực.
