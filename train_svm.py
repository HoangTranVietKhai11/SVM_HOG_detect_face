import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

# Tạo thư mục models nếu chưa có
os.makedirs("models", exist_ok=True)
PROCESSED_DIR = "data/processed"

def train_models():
    # ==========================================
    # 1. Huấn luyện SVM Model 1 (Identity)
    # ==========================================
    print("--- Đang huấn luyện Model Danh Tính ---")
    
    # Load dataset đã lưu từ quá trình tiền xử lý
    X_id = np.load(os.path.join(PROCESSED_DIR, "X_identity.npy"))
    y_id = np.load(os.path.join(PROCESSED_DIR, "y_identity.npy"))

    # Chia tập train/test theo tỷ lệ 80/20 (Khớp báo cáo)
    X_train_id, X_test_id, y_train_id, y_test_id = train_test_split(
        X_id, y_id, test_size=0.2, random_state=42, stratify=y_id
    )

    model_identity = LinearSVC(class_weight='balanced', random_state=42, max_iter=3000)
    model_identity.fit(X_train_id, y_train_id)
    
    # Đánh giá Accuracy trên tập Test
    preds_id = model_identity.predict(X_test_id)
    print("Test accuracy (Identity):", accuracy_score(y_test_id, preds_id))
    print(classification_report(y_test_id, preds_id, zero_division=0))

    # Lưu model
    joblib.dump(model_identity, "models/model_identity.pkl")
    print("=> Đã lưu thành công: models/model_identity.pkl\n")

    # ==========================================
    # 2. Huấn luyện SVM Model 2 (Liveness)
    # ==========================================
    print("--- Đang huấn luyện Model Liveness (Thật/Giả) ---")
    X_live = np.load(os.path.join(PROCESSED_DIR, "X_liveness.npy"))
    y_live = np.load(os.path.join(PROCESSED_DIR, "y_liveness.npy"))

    # Chia tập train/test theo tỷ lệ 80/20
    X_train_live, X_test_live, y_train_live, y_test_live = train_test_split(
        X_live, y_live, test_size=0.2, random_state=42, stratify=y_live
    )

    model_liveness = LinearSVC(class_weight='balanced', random_state=42, max_iter=3000)
    model_liveness.fit(X_train_live, y_train_live)

    # Đánh giá Accuracy trên tập Test
    preds_live = model_liveness.predict(X_test_live)
    print("Test accuracy (Liveness):", accuracy_score(y_test_live, preds_live))
    print(classification_report(y_test_live, preds_live, zero_division=0))

    # Lưu model
    joblib.dump(model_liveness, "models/model_liveness.pkl")
    print("=> Đã lưu thành công: models/model_liveness.pkl")

if __name__ == "__main__":
    train_models()