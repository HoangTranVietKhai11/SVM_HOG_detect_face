# HƯỚNG DẪN THUYẾT TRÌNH & ÔN TẬP VẤN ĐÁP ĐỒ ÁN HOG + SVM

Tài liệu này được biên soạn bám sát 100% vào **Tiêu chí đánh giá cuối kỳ (Rubric)** của thầy giáo. Khải và Đức hãy đọc kỹ để bảo vệ đồ án tự tin và ăn trọn điểm nhé.

---

## PHẦN 1: KỊCH BẢN THUYẾT TRÌNH (Tối ưu cho 5-10 phút)

### 1. Mở đầu & Đặt vấn đề (Tương ứng 20% điểm Sáng tạo)
**Người nói:** Khải (hoặc Đức)
> *"Chào thầy và các bạn, nhóm chúng em chọn Chủ đề 2: HOG + SVM. Thông thường bài toán này dùng để nhận diện người đi bộ (Pedestrian Detection). Tuy nhiên, để tăng tính ứng dụng thực tế, nhóm em đã **mở rộng đề tài** thành: Xây dựng hệ thống điểm danh nhận diện danh tính và tích hợp chống giả mạo (Anti-Spoofing) bằng hình ảnh."*

### 2. Trình bày Lý thuyết (Tương ứng 30% điểm Cơ sở toán học)
**Lưu ý:** Thầy cấm copy nguyên văn từ AI, nên bạn hãy nói theo ý hiểu của mình.
> *"Để làm được điều này, hệ thống của tụi em hoạt động theo 3 bước:*
> 1. *Đầu tiên, ảnh được đưa qua toán tử Sobel để tính **Gradient (Độ dốc ảnh)**. Nó sẽ tìm ra các đường nét, góc cạnh của khuôn mặt bằng cách tính đạo hàm sự thay đổi ánh sáng.*
> 2. *Sau đó, tụi em chia ảnh thành các ô (Cell 8x8) và khối (Block 16x16) để tính **HOG (Histogram of Oriented Gradients)**. Nhờ kỹ thuật chuẩn hóa (Normalization) trong từng Block, đặc trưng HOG của nhóm em ít bị ảnh hưởng bởi ánh sáng chói hay tối.*
> 3. *Cuối cùng, Vector HOG (1764 chiều) được đưa vào mô hình **Linear SVM**. SVM sẽ tìm ra một siêu phẳng (Hyperplane) tối ưu để phân loại đâu là Khải, đâu là Đức, và đâu là người lạ."*

### 3. Demo sản phẩm (Tương ứng 40% điểm Chất lượng ứng dụng)
> *"Sau đây em xin Demo hệ thống thực tế trên dữ liệu tụi em tự thu thập."*
*   **Demo 1 (Webcam):** Bật Camera (Tab 2). Cả Khải và Đức bước vào. Hệ thống vẽ khung Xanh báo đúng tên Khải và Đức.
*   **Demo 2 (Phát hiện lừa đảo):** Đưa một bức ảnh của Khải (trên điện thoại) ra trước Camera. Hệ thống sẽ phát hiện độ bóng/lóa của màn hình, vẽ khung Cam báo "GIA MAO".
*   **Demo 3 (Video nhiều người):** Bật Tab 3, tải lên Video đám đông đi bộ. Hệ thống vẽ khung Tím (HUMAN BODY) bao quanh toàn bộ người đi bộ (đáp ứng đúng yêu cầu cơ bản của đề).

---

## PHẦN 2: NGÂN HÀNG CÂU HỎI VẤN ĐÁP (10% Liêm chính học thuật)
Giảng viên sẽ hỏi để kiểm tra xem bạn có hiểu code không hay chỉ nhờ AI làm. Bạn hãy học thuộc các ý này:

### Câu 1: Tại sao nhóm lại chọn chuẩn hóa ảnh về 64x64 trước khi rút trích HOG?
**Trả lời:** Dạ thưa thầy, đầu vào của SVM yêu cầu các vector đặc trưng phải có số chiều bằng nhau. Thuật toán HOG phân chia ảnh theo Pixel (Ví dụ Block 16x16, Cell 8x8). Nếu ảnh đầu vào kích thước không đồng nhất, thì số lượng Block sinh ra sẽ khác nhau, dẫn đến Vector HOG có độ dài ngắn khác nhau. Việc cố định 64x64 giúp tất cả các vector HOG sinh ra đều có đúng 1764 chiều, đảm bảo tính đồng bộ cho ma trận của SVM ạ.

### Câu 2: Ý nghĩa vật lý của Magnitude (Biên độ) và Orientation (Hướng) là gì?
**Trả lời:** 
*   **Magnitude:** Đại diện cho độ sắc nét (độ mạnh) của cạnh (Edge). Điểm ảnh nào chuyển từ đen sang trắng càng gắt thì Magnitude càng lớn.
*   **Orientation:** Chỉ ra hướng vuông góc với cạnh đó. Trong HOG, các hướng này được gom vào 9 Bins (từ 0-180 độ) để tạo thành biểu đồ (Histogram), giúp mô tả hình dáng của đối tượng.

### Câu 3: Kích thước Cell và Block trong HOG của nhóm là bao nhiêu? Tại sao lại chọn số đó?
**Trả lời:** 
Dạ nhóm em thiết lập thông số mặc định như sau: `Cell size = 8x8`, `Block size = 16x16`, `Stride = 8` (dịch chuyển nửa block), `Bins = 9`. 
*   **Cell 8x8** đủ nhỏ để bắt được chi tiết đường nét của khuôn mặt.
*   **Block 16x16 (gồm 4 cells)** dùng để chuẩn hóa (Normalize) theo cụm. Việc này rất quan trọng để khử nhiễu do bóng râm hoặc ánh sáng môi trường thay đổi.

### Câu 4: Làm sao hệ thống của nhóm phát hiện được đâu là Người Lạ (Unknown)?
**Trả lời:** Thưa thầy, ngoài việc thu thập ảnh của 2 thành viên nhóm, tụi em đã tải 50 bức ảnh (Negative Samples) của nhiều người lạ mặt khác nhau trên mạng và gán nhãn là 'Unknown'. SVM được huấn luyện đa lớp (Multi-class). Nhờ vậy, nếu một đặc trưng khuôn mặt không nằm trong vùng phân bố của Khải hay Đức, SVM sẽ xếp nó vào lớp Người Lạ ạ.

### Câu 5: Em có dùng công cụ AI (ChatGPT, Copilot) để hỗ trợ không? Nhóm đã tự làm phần nào?
*(Đây là câu gài bẫy theo mục 4 của Rubric, hãy trả lời trung thực và tự tin)*
**Trả lời:** Dạ thưa thầy, theo quy định, tụi em có dùng AI để hỗ trợ tìm cú pháp hàm của `cv2` (ví dụ: cú pháp lấy `cv2.HOGDescriptor`), debug lỗi khi kết xuất Camera lên Streamlit và sinh mã LaTeX cho công thức toán trong báo cáo. 
Tuy nhiên, **toàn bộ luồng Logic xử lý** từ việc nghĩ ra thuật toán Anti-spoofing, tự quay 100 ảnh để tạo Dataset, hay quyết định việc dùng hàm `cv2.equalizeHist()` để tiền xử lý ánh sáng trước khi quét Haar Cascade đều là do nhóm tự phân tích và thử nghiệm thủ công để chốt thông số ạ.

---

## PHẦN 3: NHIỆM VỤ CẦN CHUẨN BỊ TRƯỚC HÔM THUYẾT TRÌNH
1. **Kiểm tra file Báo Cáo (`Bao_cao_Cuoi_ky.md`):** Hãy mở file ra và sửa lại lời văn trong CHƯƠNG 2 theo văn phong của bạn. Đừng để câu chữ quá "máy móc" vì thầy cấm copy nguyên văn bài của AI.
2. **Dữ liệu Video Demo:** Chuẩn bị sẵn 1 file video (MP4) có người đi bộ để demo thẳng vào Tab 3.
3. **Môi trường:** Đảm bảo cài đặt đủ thư viện và lệnh `streamlit run app.py` chạy mượt mà không báo lỗi.
