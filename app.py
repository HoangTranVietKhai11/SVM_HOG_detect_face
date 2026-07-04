import streamlit as st
import cv2
import numpy as np
from predict import process_and_predict

# Tiêu đề Web App
st.title("🎭 Ứng dụng Điểm danh Khuôn mặt")
st.subheader("Tích hợp Anti-Spoofing (HOG + SVM)")

# Cho phép upload ảnh
uploaded_file = st.file_uploader("Upload ảnh khuôn mặt để kiểm tra...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc file ảnh người dùng upload thành ma trận numpy (OpenCV format)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    st.write("Đang phân tích hình ảnh...")
    
    # Gọi hàm xử lý từ predict.py
    results = process_and_predict(img)
    
    if len(results) == 0:
        st.warning("Không phát hiện thấy khuôn mặt nào trong ảnh!")
    else:
        # Vẽ Bounding Box và Nhãn
        for res in results:
            x, y, w, h = res['box']
            name = res['name']
            liveness = res['liveness']
            
            # Khung Xanh lá cho ảnh Thật, Đỏ cho Giả mạo
            color = (0, 255, 0) if liveness == "Real" else (0, 0, 255)
            
            # Gộp nhãn hiển thị: VD "Khai (Real)" hoặc "Duc (Spoof)"
            label = f"{name} ({liveness})"
            
            # Vẽ hình chữ nhật và in chữ lên ảnh gốc
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # OpenCV dùng hệ màu BGR, Streamlit hiển thị bằng RGB nên cần chuyển đổi
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Hiển thị kết quả
        st.success(f"Đã phát hiện {len(results)} khuôn mặt!")
        st.image(img_rgb, caption="Kết quả nhận diện", use_column_width=True)