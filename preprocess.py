import os
import cv2
import numpy as np
# Import hàm cắt mặt từ file của Khải
from face_detector import detect_and_crop

# Cấu hình tham số HOG theo yêu cầu của đồ án
# winSize=(64,64), blockSize=(16,16), blockStride=(8,8), cellSize=(8,8), nbins=9
hog = cv2.HOGDescriptor(
    _winSize=(64, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

# Đường dẫn dataset
dataset_paths = {
    "khai": "dataset/real/khai",
    "duc": "dataset/real/duc",
    "spoof": "dataset/spoof"
}

def extract_features():
    X_identity, y_identity = [], []
    X_liveness, y_liveness = [], []

    for label_name, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"Cảnh báo: Không tìm thấy thư mục {path}")
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            
            # Cắt khuôn mặt và resize về 64x64
            face, _ = detect_and_crop(img_path)
            
            if face is not None:
                # Trích xuất đặc trưng HOG
                h_features = hog.compute(face).flatten()

                # Gán nhãn cho bài toán Liveness (1: Real, 0: Spoof)
                if label_name in ["khai", "duc"]:
                    X_liveness.append(h_features)
                    y_liveness.append(1) # Real
                    
                    # Gán nhãn cho bài toán Identity (chỉ xét ảnh thật)
                    X_identity.append(h_features)
                    y_identity.append(0 if label_name == "khai" else 1) # 0: Khải, 1: Đức
                else:
                    X_liveness.append(h_features)
                    y_liveness.append(0) # Spoof

    # Lưu dữ liệu ra file .npy
    np.save("X_train_identity.npy", np.array(X_identity))
    np.save("y_train_identity.npy", np.array(y_identity))
    np.save("X_train_liveness.npy", np.array(X_liveness))
    np.save("y_train_liveness.npy", np.array(y_liveness))
    
    print("Đã trích xuất và lưu xong dữ liệu!")

if __name__ == "__main__":
    extract_features()