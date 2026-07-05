import cv2
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from hog_extractor import extract_hog
from face_detector import detect_and_crop

DATASET_PATH = "dataset"

# Nhãn cần augment (nhãn thiểu số)
AUGMENT_LABELS = {"person_a", "person_b"}


def augment_face(face):
    """Tạo nhiều phiên bản augmented từ một ảnh khuôn mặt."""
    results = [face]

    # Lật ngang
    results.append(cv2.flip(face, 1))

    # Xoay ±10 độ
    h, w = face.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face, M, (w, h))
        results.append(rotated)

    # Tăng/giảm độ sáng
    results.append(cv2.convertScaleAbs(face, alpha=1.2, beta=20))
    results.append(cv2.convertScaleAbs(face, alpha=0.8, beta=-20))

    return results  # 6 ảnh mỗi gốc

def load_dataset():
    X = []
    y = []
    
    labels = os.listdir(DATASET_PATH)
    print("Nhãn tìm thấy:", labels)
    
    for label in labels:
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(folder):
            continue
        
        files = os.listdir(folder)
        print(f"Đang xử lý nhãn '{label}': {len(files)} ảnh")
        
        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            img_path = os.path.join(folder, filename)
            
            # Detect và crop khuôn mặt
            face, _ = detect_and_crop(img_path)
            
            if face is None:
                print(f"  Bỏ qua (không detect được mặt): {filename}")
                continue
            
            # Augment nếu là nhãn thiểu số, ngược lại chỉ dùng ảnh gốc
            faces = augment_face(face) if label in AUGMENT_LABELS else [face]

            for f in faces:
                hog_vec = extract_hog(f)
                X.append(hog_vec)
                y.append(label)
    
    return np.array(X), np.array(y)


def train_and_save():
    print("=== BẮT ĐẦU LOAD DATASET ===")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("Không load được ảnh nào — kiểm tra lại cấu trúc thư mục")
        return
    
    print(f"\nTổng ảnh load được: {len(X)}")
    print(f"Phân bố nhãn: { {label: list(y).count(label) for label in set(y)} }")
    
    # Chia train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)} ảnh | Test: {len(X_test)} ảnh")
    
    # Train SVM
    print("\n=== BẮT ĐẦU TRAIN SVM ===")
    model = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)
    
    # Đánh giá
    y_pred = model.predict(X_test)
    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    print(classification_report(y_test, y_pred))
    
    # Lưu model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_identity.pkl")
    print("Đã lưu model vào models/model_identity.pkl")


if __name__ == "__main__":
    train_and_save()