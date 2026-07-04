import cv2 
import numpy as np 
import joblib
import os 

from face_detector import detect_and_crop
from hog_extractor import extract_hog

MODEL_IDENTITY_PATH = "models/model_identity.pkl"
MODEL_LIVENESS_PATH = "models/model_liveness.pkl"

def load_models():
    if not os.path.exists(MODEL_IDENTITY_PATH):
        raise FileNotFoundError(f"Chưa có model: {MODEL_IDENTITY_PATH} — chờ Đức train xong gửi file .pkl")
    if not os.path.exists(MODEL_LIVENESS_PATH):
        raise FileNotFoundError(f"Chưa có model: {MODEL_LIVENESS_PATH} — chờ Đức train xong gửi file .pkl")
    
    model_identity = joblib.load(MODEL_IDENTITY_PATH)
    model_liveness = joblib.load(MODEL_LIVENESS_PATH)
    return model_identity, model_liveness

def predict(image_path):
    #buoc1: detect va crop khuon mat
    face, original_img = detect_and_crop(image_path)

    if face is None:
        return {
            "status": "error",
            "message": "Không detect được khuôn mặt"
        }
    # buoc 2: trich xuat HOG vector
    hog_vector = extract_hog(face)
    hog_vector = hog_vector.reshape(1,-1) # SVM can input shape (1, n_features)

    # buoc 3: load model va predict
    model_identity, model_liveness = load_models()

    identity = model_identity.predict(hog_vector)[0]
    liveness = model_liveness.predict(hog_vector)[0]
    
    #buoc 4: ve bounding box len anh goc

    result_img = draw_result(original_img, identity, liveness)

    return {
        "status": "ok",
        "identity": identity,
        "liveness": liveness,
        "result_img": result_img
    }

def draw_result(img, identity, liveness):
    # Detect lại để lấy tọa độ box vẽ lên ảnh gốc
    import cv2
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    
    result = img.copy()
    
    for (x, y, w, h) in faces:
        # Màu box: xanh lá = Thật, đỏ = Giả mạo
        color = (0, 255, 0) if liveness == "real" else (0, 0, 255)
        
        # Vẽ bounding box
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        
        # Nhãn hiển thị
        label = f"{identity.upper()} | {'THAT' if liveness == 'real' else 'GIA MAO'}"
        cv2.putText(result, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result
if __name__ == "__main__":
    result = predict("ten_anh_cua_ban.jpg")  # sửa tên ảnh
    
    if result["status"] == "ok":
        print("Danh tính:", result["identity"])
        print("Liveness:", result["liveness"])
        cv2.imwrite("ket_qua.jpg", result["result_img"])
        print("Đã lưu ảnh kết quả vào ket_qua.jpg")
    else:
        print("Lỗi:", result["message"])
    