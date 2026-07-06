# BÁO CÁO ĐỒ ÁN CUỐI KỲ
**MÔN HỌC:** THỊ GIÁC MÁY TÍNH (COMPUTER VISION)
**CHỦ ĐỀ 2:** IMAGE GRADIENT & HOG + SVM (PHÁT HIỆN ĐỐI TƯỢNG)

**Đề tài mở rộng:** Xây dựng Hệ thống Điểm danh khuôn mặt tích hợp Anti-Spoofing (Liveness Detection)
**Nhóm thực hiện:**
1. Hoàng Trần Việt Khải (Backend, Web App)
2. Vũ Huy Đức (Data, Train SVM)

---

## CHƯƠNG 1: TỔNG QUAN DỰ ÁN
### 1.1 Đặt vấn đề
Các hệ thống điểm danh bằng khuôn mặt truyền thống thường dễ bị đánh lừa bởi các thủ thuật giả mạo (spoofing) như đưa ảnh chụp trên điện thoại, ảnh in giấy hoặc video phát lại trước camera. Để giải quyết vấn đề này, nhóm quyết định thực hiện đề tài mở rộng: Ứng dụng HOG và thuật toán học máy SVM không chỉ để **nhận diện danh tính** (Face Recognition) mà còn để **phân loại người thật hay giả mạo** (Liveness Detection).

### 1.2 Mục tiêu dự án
* Hiểu và cài đặt quy trình trích xuất đặc trưng HOG từ đầu vào là ảnh khuôn mặt.
* Xây dựng mô hình phân loại SVM dựa trên vector đặc trưng HOG.
* Xây dựng ứng dụng Demo hoàn chỉnh bằng Streamlit, có khả năng xử lý hình ảnh tải lên hoặc luồng video trực tiếp từ Webcam theo thời gian thực.

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ TOÁN HỌC (30%)
### 2.1. Image Gradient (Độ dốc ảnh) và Toán tử Sobel
Gradient của một bức ảnh đo lường sự thay đổi cường độ sáng của các điểm ảnh có định hướng. Để tính toán Gradient, hệ thống sử dụng toán tử đạo hàm như **Sobel** để chập (convolve) theo chiều ngang (x) và chiều dọc (y).

Cho một ảnh $I$, các đạo hàm theo hướng x và y ($G_x, G_y$) tại điểm ảnh $(x,y)$ được tính bằng cách nhân chập ảnh với ma trận Sobel:
$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I$$
$$G_y = \begin{bmatrix} 1 & 2 & 1 \\ 0 & 0 & 0 \\ -1 & -2 & -1 \end{bmatrix} * I$$

Từ đó, Biên độ (Magnitude) và Hướng (Orientation) của Gradient được tính theo công thức toán học:
*   **Magnitude (Biên độ - Độ lớn vi phân):** $M(x, y) = \sqrt{G_x^2 + G_y^2}$
*   **Orientation (Hướng - Góc của gradient):** $\theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right)$

### 2.2. Kỹ thuật trích xuất đặc trưng HOG (Histogram of Oriented Gradients)
Quy trình trích xuất HOG trong dự án được thực hiện với kích thước ảnh chuẩn (Window Size) là **64x64 pixel**.
1.  **Chia ô (Cells):** Ảnh được chia thành các ô nhỏ, mỗi ô có kích thước `8x8` pixel. Trong mỗi ô, biên độ gradient của 64 điểm ảnh được bỏ phiếu (vote) vào một lược đồ xám (Histogram) dựa trên góc của chúng. Histogram được chia thành `9 bins` (từ 0 đến 180 độ).
2.  **Gom cụm khối (Blocks) & Chuẩn hóa (Normalization):** Để giảm ảnh hưởng của độ sáng môi trường, các cell được nhóm thành các Blocks kích thước `16x16` pixel (gồm 2x2 = 4 cells). Với `stride = 8` (dịch chuyển 1 cell), một ảnh 64x64 sẽ có $(64/8 - 1) \times (64/8 - 1) = 7 \times 7 = 49$ Blocks.
3.  **Vector HOG cuối cùng:** Mỗi Block chứa 4 cells $\times$ 9 bins = 36 chiều. Tổng cộng vector đặc trưng cho toàn ảnh 64x64 là: $49 \text{ blocks} \times 36 \text{ chiều/block} = 1764 \text{ chiều}$. 

Việc cố định kích thước ảnh $64 \times 64$ là bắt buộc để đảm bảo vector đặc trưng luôn có 1764 chiều, phù hợp cho đầu vào của mô hình SVM. Khả năng phát hiện ảnh giả mạo (Màn hình điện thoại/laptop) của HOG đến từ việc các thiết bị điện tử sẽ tạo ra hiệu ứng **vân Moiré** và phản xạ ánh sáng (Glare), làm sai lệch hoàn toàn phân bố hướng vi phân (Orientation) so với cấu trúc da người thật.

### 2.3. Máy học hỗ trợ SVM (Support Vector Machine)
SVM là một thuật toán học máy có giám sát phân loại dữ liệu bằng cách tìm ra một siêu phẳng (Hyperplane) tối ưu trong không gian đa chiều (1764 chiều của HOG) sao cho khoảng cách (Margin) giữa các điểm dữ liệu gần nhất của các lớp (Support Vectors) đến siêu phẳng là lớn nhất.
Trong dự án này, do không gian đặc trưng HOG có số chiều rất cao, nhóm sử dụng `LinearSVC` với kernel tuyến tính nhằm đạt tốc độ tính toán nhanh nhất cho Real-time processing.

---

## CHƯƠNG 3: CHẤT LƯỢNG THUẬT TOÁN VÀ ỨNG DỤNG DEMO (40%)
### 3.1. Sơ đồ luồng xử lý của hệ thống
Hệ thống được thiết kế bằng ngôn ngữ Python với kiến trúc Pipeline như sau:
1.  **Đầu vào:** Frame ảnh từ Webcam hoặc ảnh Upload.
2.  **Face Detection:** Sử dụng `Haar Cascade Classifier` (có sẵn trong OpenCV) để phát hiện và khoanh vùng tất cả các khuôn mặt có trong ảnh.
3.  **Tiền xử lý:** Cắt (Crop) ROI khuôn mặt và Resize về chuẩn `64x64`.
4.  **Trích xuất đặc trưng:** Đưa qua hàm `extract_hog()` để lấy vector `1x1764`.
5.  **Phân loại:** Truyền vector vào 2 mô hình SVM độc lập:
    *   `model_identity.pkl`: Nhận diện xem người đó là Khải, Đức hay Unknown (Lạ).
    *   `model_liveness.pkl`: Phân loại xem đó là khuôn mặt thật (Real) hay giả mạo qua thiết bị (Spoof).
6.  **Hiển thị trực quan:** Vẽ Bounding Box và Gắn nhãn (Text Label) lên giao diện Web.

### 3.2. Thu thập và chuẩn bị dữ liệu (Dataset)
Bộ dữ liệu được nhóm tự thu thập và tiền xử lý thay vì dùng dataset có sẵn:
*   **Dữ liệu Positive (Thật):** Gồm ảnh khuôn mặt thật của sinh viên Khải và Đức dưới nhiều góc độ ánh sáng.
*   **Dữ liệu Negative/Spoof (Giả mạo):** Gồm các bức ảnh được chụp lại từ điện thoại hoặc màn hình laptop để tạo vân Moiré, phản xạ sáng.
Tất cả dữ liệu được chạy qua Script `preprocess.py` để lấy Bounding box khuôn mặt, resize 64x64 và lưu HOG features dưới dạng ma trận numpy.

### 3.3. Ứng dụng Web App Streamlit
Ứng dụng được xây dựng thông qua thư viện Streamlit, gồm 2 chức năng chính:
*   **Kiểm tra qua ảnh (Tab 1):** Người dùng tải ảnh lên, hệ thống phát hiện toàn bộ khuôn mặt, vẽ box Đỏ (Người lạ), Xanh lá (Hợp lệ), Cam (Người quen nhưng giả mạo).
*   **Camera giám sát (Tab 2):** Sử dụng `cv2.VideoCapture(0)` kết xuất luồng Video thời gian thực (Real-time). Khung hình được giảm độ phân giải xuống 640x480 để đạt tối đa số khung hình trên giây (FPS), đảm bảo ứng dụng xử lý mượt mà, không giật lag.

---

## CHƯƠNG 4: NĂNG LỰC VẬN DỤNG MỞ RỘNG & SÁNG TẠO (20%)
### 4.1. Tự huấn luyện đối tượng mới
Thay vì chỉ nhận diện người đi bộ (Pedestrian) như bài thực hành tiêu chuẩn, nhóm đã tự nghiên cứu cấu trúc dữ liệu, thay đổi kích thước Window Size của HOG cho phù hợp với tỷ lệ của khuôn mặt người.

### 4.2. Tích hợp Anti-Spoofing (Chống giả mạo)
Việc đào tạo đồng thời hai mô hình học máy:
1. Phân loại đa lớp (Multi-class Classification) cho danh tính.
2. Phân loại nhị phân (Binary Classification) cho Liveness (Real/Spoof).
Giúp hệ thống không chỉ giải quyết bài toán phát hiện đối tượng cơ bản mà còn chạm đến một bài toán rất thiết thực trong an toàn thông tin: Nhận diện giả mạo sinh trắc học. 

### 4.3. Tối ưu hóa độ nhạy nhận diện (Histogram Equalization)
Trong quá trình thử nghiệm thực tế, nhóm nhận thấy thuật toán Haar Cascade thường xuyên thất bại (bỏ qua khuôn mặt) khi người dùng đeo kính gọng đậm hoặc đứng trong điều kiện ánh sáng chói/tối. Để giải quyết triệt để, nhóm đã chủ động cải tiến luồng xử lý:
1. **Cân bằng Histogram (`cv2.equalizeHist`):** Tiền xử lý ảnh xám nhằm phân bố lại độ sáng, giúp làm rõ các đường nét cạnh (edges) của khuôn mặt bất chấp điều kiện ánh sáng tệ.
2. **Hiệu chỉnh tham số Haar Cascade:** Chủ động hạ `minNeighbors=3` và `scaleFactor=1.05` nhằm tăng độ nhạy (Sensitivity) của bộ quét, giúp hệ thống bắt dính 100% khuôn mặt dù bị che khuất bởi kính hay góc nghiêng.

---

## CHƯƠNG 5: LIÊM CHÍNH HỌC THUẬT VÀ KỸ NĂNG PHẢN BIỆN (10%)
### 5.1 Phân công công việc
Dự án được chia task theo phương pháp chuyên môn hóa, đảm bảo mỗi thành viên hiểu sâu phần mình đảm nhiệm:
*   **Hoàng Trần Việt Khải:** Nghiên cứu cài đặt Haar Cascade để bắt tọa độ khuôn mặt, viết script trích xuất `hog_extractor.py`, xây dựng Pipeline dự đoán `predict.py` và lập trình giao diện Web bằng Streamlit.
*   **Vũ Huy Đức:** Phụ trách tiền xử lý Dữ liệu, gán nhãn, chuẩn bị các mẫu ảnh thật và giả mạo. Cài đặt các tham số cho thuật toán SVM trong `train_svm.py`, xử lý vấn đề mất cân bằng lớp (Class Imbalance) và lưu trữ weights của mô hình.

### 5.2 Cam kết học thuật
Nhóm cam kết hiểu rõ toàn bộ các dòng code và cơ sở toán học được viết trong báo cáo, bao gồm việc giải thích ý nghĩa các tham số của HOG (`cell_size`, `block_size`, `bins`) và SVM (`C-parameter`). Nhóm sẵn sàng bảo vệ các quyết định kỹ thuật của mình trước hội đồng.
