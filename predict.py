import cv2
import joblib
import numpy as np

# Load 2 mô hình SVM của Đức
try:
    model_identity = joblib.load('models/model_identity.pkl')
    model_liveness = joblib.load('models/model_liveness.pkl')
except FileNotFoundError:
    print("Chưa tìm thấy file .pkl. Hãy chắc chắn đã chạy train_svm.py!")

# Load Haar Cascade và cấu hình HOG chuẩn
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
hog = cv2.HOGDescriptor(_winSize=(64, 64), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)

def process_and_predict(image):
    # Chuyển ảnh màu sang xám để detect
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    
    results = []
    
    # Duyệt qua từng khuôn mặt phát hiện được
    for (x, y, w, h) in faces:
        # Cắt và resize về 64x64
        face_roi = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64, 64))
        
        # Trích xuất HOG
        features = hog.compute(face_resized).flatten().reshape(1, -1)
        
        # Dự đoán Danh tính và Thật/Giả
        id_pred = model_identity.predict(features)[0] 
        live_pred = model_liveness.predict(features)[0] 
        
        # Mapping nhãn theo quy ước của hàm train_svm.py
        name = "Khai" if id_pred == 0 else "Duc"
        liveness = "Real" if live_pred == 1 else "Spoof"
        
        # Lưu lại tọa độ và kết quả để giao diện vẽ
        results.append({
            "box": (x, y, w, h), 
            "name": name, 
            "liveness": liveness
        })
        
    return results