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

### Bước 2: Thu thập thêm dữ liệu (Tuỳ chọn)
Nếu bạn muốn hệ thống nhận diện thêm khuôn mặt mới, hãy chạy script thu thập ảnh tự động:
```bash
python capture_faces.py
```
Sau đó làm theo hướng dẫn trên màn hình (nhấn phím 1, 2, 3 tương ứng để thêm khuôn mặt, nhấn `q` để thoát). Dữ liệu sẽ tự động lưu vào `data/raw/`.

### Bước 3: Tiền xử lý dữ liệu và trích xuất đặc trưng HOG
Trước khi huấn luyện, cần đưa ảnh gốc qua module cắt mặt, tăng cường dữ liệu (augmentation) và trích xuất thành vector HOG:
```bash
python preprocess.py
```
Hệ thống sẽ chạy và tự động lưu các ma trận `.npy` vào thư mục `data/processed/`.

### Bước 4: Huấn luyện mô hình SVM
Thực hiện huấn luyện cả 2 mô hình (Identity và Liveness) dựa trên dữ liệu đã được tiền xử lý:
```bash
python train_svm.py
```
Sau khi huấn luyện thành công, hệ thống sẽ in ra màn hình độ chính xác (Accuracy) và tự động lưu mô hình vào thư mục `models/`.

### Bước 5: Chạy ứng dụng Giao diện Web (Streamlit)
Cuối cùng, để trải nghiệm hệ thống điểm danh và giám sát camera:
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở trang web (mặc định tại `localhost:8501`). Bạn có thể chuyển đổi giữa 2 Tab (Tải ảnh tĩnh và Camera giám sát thời gian thực) để kiểm tra kết quả.
