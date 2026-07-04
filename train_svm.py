import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Tạo thư mục models nếu chưa có
os.makedirs("models", exist_ok=True)

def train_models():
    # ==========================================
    # 1. Huấn luyện SVM Model 1 (Identity)
    # ==========================================
    print("--- Đang huấn luyện Model Danh Tính ---")
    X_id = np.load("X_train_identity.npy")
    y_id = np.load("y_train_identity.npy")

    model_identity = LinearSVC(class_weight='balanced', random_state=42)
    model_identity.fit(X_id, y_id)
    
    # Đánh giá Accuracy
    preds_id = model_identity.predict(X_id)
    print("Độ chính xác (Accuracy):", accuracy_score(y_id, preds_id))
    print(classification_report(y_id, preds_id, target_names=["Khai", "Duc"]))

    # Lưu model
    joblib.dump(model_identity, "models/model_identity.pkl")
    print("=> Đã lưu thành công: models/model_identity.pkl\n")

    # ==========================================
    # 2. Huấn luyện SVM Model 2 (Liveness)
    # ==========================================
    print("--- Đang huấn luyện Model Liveness (Thật/Giả) ---")
    X_live = np.load("X_train_liveness.npy")
    y_live = np.load("y_train_liveness.npy")

    model_liveness = LinearSVC(class_weight='balanced', random_state=42)
    model_liveness.fit(X_live, y_live)

    # Đánh giá Accuracy
    preds_live = model_liveness.predict(X_live)
    print("Độ chính xác (Accuracy):", accuracy_score(y_live, preds_live))
    print(classification_report(y_live, preds_live, target_names=["Spoof", "Real"]))

    # Lưu model
    joblib.dump(model_liveness, "models/model_liveness.pkl")
    print("=> Đã lưu thành công: models/model_liveness.pkl")

if __name__ == "__main__":
    train_models()