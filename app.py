import streamlit as st
import cv2
import numpy as np
from PIL import Image
from predict import process_and_predict

st.set_page_config(page_title="Hệ thống Điểm danh", layout="wide")

st.title("HỆ THỐNG ĐIỂM DANH THÔNG MINH")
st.subheader("Hệ thống An ninh & Kiểm soát ra vào")
st.markdown("---")

# Tạo 3 tab: Upload ảnh, Camera trực tiếp và Video
tab1, tab2, tab3 = st.tabs(["Kiểm tra qua ảnh", "Camera giám sát", "Kiểm tra qua Video"])

# ================= TAB 1: UPLOAD ẢNH =================
with tab1:
    uploaded_file = st.file_uploader(
        "Upload ảnh khuôn mặt để kiểm tra...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Không đọc được ảnh. Vui lòng thử file khác.")
        else:
            st.write("Đang phân tích hình ảnh...")

            try:
                results = process_and_predict(img)

                if len(results) == 0:
                    st.warning("Không phát hiện khuôn mặt.")
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col2:
                        with st.container(border=True):
                            st.image(pil_img, caption="Ảnh gốc", use_column_width=True)
                else:
                    result_img = img.copy()
                    for res in results:
                        x, y, w, h = res['box']
                        
                        # Unknown = Người lạ, ngược lại là tên người được cấp quyền
                        if res['name'].lower() == 'unknown':
                            name_display = "KHONG HOP LE"
                            box_color = (0, 0, 255) # Đỏ cho người lạ
                        else:
                            name_display = res['name'].upper()
                            # Xanh lá nếu là real, Cam nếu là Spoof (nhưng được nhận diện danh tính)
                            box_color = (0, 255, 0) if res['liveness'] == "Real" else (0, 165, 255)

                        cv2.rectangle(result_img, (x, y), (x+w, y+h), box_color, 2)
                        label = f"{name_display} | {res['liveness']}"
                        cv2.putText(result_img, label, (x, max(y - 10, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                    st.success(f"Đã phát hiện {len(results)} đối tượng.")

                    for i, res in enumerate(results):
                        is_stranger = res['name'].lower() == 'unknown'
                        role_icon = "CẢNH BÁO: ĐỐI TƯỢNG CHƯA ĐĂNG KÝ" if is_stranger else f"XÁC THỰC THÀNH CÔNG: {res['name'].upper()}"
                        liveness_text = "Hợp lệ (Người thật)" if res['liveness'] == "Real" else "Cảnh báo giả mạo"
                        
                        st.write(
                            f"**Đối tượng #{i+1}** — "
                            f"{role_icon} | "
                            f"Trạng thái: **{liveness_text}**"
                        )

                    cv2.imwrite("temp_result.jpg", result_img)
                    st.markdown("---")
                    st.markdown("### KHUNG KẾT QUẢ")
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col2:
                        with st.container(border=True):
                            st.image("temp_result.jpg", caption="Kết quả nhận diện", use_column_width=True)

            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                st.exception(e)

# ================= TAB 2: CAMERA TRỰC TIẾP =================
with tab2:
    st.markdown("""
    **Hướng dẫn:** Kích hoạt Camera giám sát để bắt đầu kiểm tra an ninh. 
    Hệ thống sẽ quét liên tục để nhận diện danh tính hợp lệ.
    """)
    
    col_cam, col_ref = st.columns([2, 1])
    
    with col_ref:
        st.markdown("### CƠ SỞ DỮ LIỆU")
        st.info("Trạng thái truy xuất:")
        # Tạo khung trống để liên tục cập nhật ảnh (Slideshow)
        ref_placeholder = st.empty()

    with col_cam:
        run_camera = st.checkbox("Kích hoạt Camera giám sát")
        FRAME_WINDOW = st.image([])

        if run_camera:
            # Lấy trước danh sách toàn bộ ảnh trong database để làm hiệu ứng "tìm kiếm"
            import glob
            import time
            all_db_images = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                all_db_images.extend(glob.glob(f"dataset/person_*/{ext}"))
                
            # 0 là ID của webcam mặc định trên laptop
            cap = cv2.VideoCapture(0)
            
            # Hạ độ phân giải xuống 640x480 để tăng TỐC ĐỘ (FPS) cực nhanh
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                st.error("Không thể kết nối với Camera! Hãy kiểm tra quyền truy cập Camera của Windows.")
            
            # Khởi tạo mô hình HOG nhận diện người mặc định của OpenCV
            hog_body = cv2.HOGDescriptor()
            hog_body.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Lỗi đọc dữ liệu từ Camera.")
                    break
                    
                # Lật khung hình (như gương) cho dễ nhìn
                frame = cv2.flip(frame, 1)

                # Phân tích khuôn mặt
                results = process_and_predict(frame)
                
                matched_person = None
                
                for res in results:
                    x, y, w, h = res['box']
                    name = res['name']
                    liveness = res['liveness']
                    
                    # Logic hiển thị giống Tab 1
                    if name.lower() == 'unknown':
                        name_display = "KHONG HOP LE"
                        box_color = (0, 0, 255) # Đỏ
                    else:
                        name_display = name.upper()
                        box_color = (0, 255, 0) if liveness == "Real" else (0, 165, 255)
                        
                        # Ghi nhận nếu tìm thấy người được cấp quyền
                        if liveness == "Real":
                            matched_person = name
                            
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                    label = f"{name_display} | {liveness}"
                    cv2.putText(frame, label, (x, max(y - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                # Tự động nhận diện cơ thể người (Pedestrian) bằng HOG
                (boxes, weights) = hog_body.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
                for (x, y, w, h) in boxes:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    cv2.putText(frame, "HUMAN BODY (HOG)", (x, max(y - 10, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # ======= CẬP NHẬT KHUNG ẢNH BÊN PHẢI =======
                if matched_person:
                    # Nếu quét trúng người quen -> Dừng ảnh lại ở người đó (Lock-on)
                    folder_name = matched_person.lower().replace("persona", "person_a").replace("personb", "person_b")
                    matched_imgs = glob.glob(f"dataset/{folder_name}/*.*")
                    with ref_placeholder.container():
                        st.success(f"XÁC THỰC THÀNH CÔNG: {matched_person.upper()}")
                        if matched_imgs:
                            st.image(matched_imgs[0], caption="Ảnh gốc lưu trong CSDL")
                else:
                    # Nếu không thấy ai hoặc là Người Lạ -> Đảo liên tục ảnh database để giả lập đang tìm kiếm
                    with ref_placeholder.container():
                        if len(all_db_images) > 0:
                            # Thay đổi ảnh mỗi 0.2 giây
                            idx = int(time.time() * 5) % len(all_db_images)
                            st.info("Đang quét dữ liệu nhận diện...")
                            st.image(all_db_images[idx])

                # Chuyển BGR (OpenCV) sang RGB (Streamlit)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)
                
            else:
                # Khi bỏ tick checkbox
                cap.release()
                st.write("Camera đã được tắt.")

# ================= TAB 3: KIỂM TRA QUA VIDEO =================
with tab3:
    st.markdown("""
    **Hướng dẫn:** Tải lên một đoạn video để hệ thống tự động quét, điểm danh khuôn mặt và cơ thể.
    """)
    uploaded_video = st.file_uploader("Upload video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        # Khởi tạo mô hình HOG nhận diện người
        hog_body = cv2.HOGDescriptor()
        hog_body.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        stop_btn = st.button("Dừng Video")
        
        while vf.isOpened() and not stop_btn:
            ret, frame = vf.read()
            if not ret:
                st.success("Đã phát xong video!")
                break
                
            # Resize để xử lý nhanh hơn (tối đa 800px chiều rộng)
            h, w = frame.shape[:2]
            if w > 800:
                scale = 800 / w
                frame = cv2.resize(frame, (800, int(h * scale)))
                
            # Phân tích khuôn mặt
            results = process_and_predict(frame)
            for res in results:
                x, y, w_box, h_box = res['box']
                name = res['name']
                liveness = res['liveness']
                
                if name.lower() == 'unknown':
                    name_display = "KHONG HOP LE"
                    box_color = (0, 0, 255)
                else:
                    name_display = name.upper()
                    box_color = (0, 255, 0) if liveness == "Real" else (0, 165, 255)
                    
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), box_color, 2)
                cv2.putText(frame, f"{name_display} | {liveness}", (x, max(y - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            # Phân tích cơ thể (HOG)
            (boxes, weights) = hog_body.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            for (x, y, w_box, h_box) in boxes:
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (255, 0, 255), 2)
                cv2.putText(frame, "HUMAN BODY (HOG)", (x, max(y - 10, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb)
            
        vf.release()