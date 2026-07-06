import cv2
import numpy as np
import os
from hog_extractor import extract_hog
from augment import augment_real, make_synthetic_spoof

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Load Haar Cascade trực tiếp như trong báo cáo
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def process_identity():
    print("=== TIỀN XỬ LÝ DỮ LIỆU DANH TÍNH (IDENTITY) ===")
    X, y = [], []
    
    # Các nhãn thiểu số cần augment
    AUGMENT_LABELS = {"person_a", "person_b"}
    labels = ["person_a", "person_b", "unknown"]
    
    for label in labels:
        folder = os.path.join(RAW_DIR, label)
        if not os.path.isdir(folder):
            print(f"Không tìm thấy thư mục {folder}")
            continue
            
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Nhãn '{label}': {len(files)} ảnh")
        
        for filename in files:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Resize ảnh nếu quá lớn để chạy cho nhanh
            h, w = img.shape[:2]
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                
            # Tiền xử lý ảnh như báo cáo mô tả: cv2.equalizeHist()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Detect mặt trực tiếp
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                continue
                
            x, y_c, w, h = faces[0]
            face_roi = img[y_c:y_c+h, x:x+w]
            face_resized = cv2.resize(face_roi, (64, 64))
            
            # Augment
            faces_to_process = augment_real(face_resized) if label in AUGMENT_LABELS else [face_resized]
            
            for f in faces_to_process:
                hog_vec = extract_hog(f)
                X.append(hog_vec)
                y.append(label)
                
    X = np.array(X)
    y = np.array(y)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_identity.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "y_identity.npy"), y)
    print(f"Đã lưu X_identity.npy ({X.shape}) và y_identity.npy ({y.shape})\n")

def process_liveness():
    print("=== TIỀN XỬ LÝ DỮ LIỆU LIVENESS (THẬT/GIẢ) ===")
    X, y = [], []
    
    # Lấy tất cả ảnh thật từ các thư mục người dùng + thư mục real
    folders_to_read = ["person_a", "person_b", "unknown", "real"]
    files_with_paths = []
    
    for folder_name in folders_to_read:
        folder = os.path.join(RAW_DIR, folder_name)
        if os.path.isdir(folder):
            fs = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for f in fs:
                files_with_paths.append(os.path.join(folder, f))
                
    print(f"Tổng số ảnh thật (Real) dùng cho Liveness: {len(files_with_paths)} ảnh")
    
    for img_path in files_with_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            continue
            
        x, y_c, w, h = faces[0]
        face_roi = img[y_c:y_c+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64, 64))
        
        # Ảnh Thật: Augment
        for real_face in augment_real(face_resized):
            X.append(extract_hog(real_face))
            y.append("real")
            
        # Ảnh Giả mạo: Sinh synthetic spoof từ face_resized
        for spoof_face in make_synthetic_spoof(face_resized):
            X.append(extract_hog(spoof_face))
            y.append("spoof")
            # Tăng cường fake bằng cách lật để cân bằng số lượng
            X.append(extract_hog(cv2.flip(spoof_face, 1)))
            y.append("spoof")

    X = np.array(X)
    y = np.array(y)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_liveness.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "y_liveness.npy"), y)
    print(f"Đã lưu X_liveness.npy ({X.shape}) và y_liveness.npy ({y.shape})\n")

if __name__ == "__main__":
    process_identity()
    process_liveness()