import cv2
import numpy as np

# Load bộ nhận diện khuôn mặt Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_and_crop(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    # Resize ảnh lớn xuống max 800px để Haar Cascade chạy nhanh hơn
    MAX_DIM = 800
    h_orig, w_orig = img.shape[:2]
    scale = 1.0
    if max(h_orig, w_orig) > MAX_DIM:
        scale = MAX_DIM / max(h_orig, w_orig)
        img_small = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
    else:
        img_small = img

    # Chuyển sang ảnh xám để tăng tốc độ detect
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    # Cân bằng độ sáng (giúp nhận diện tốt hơn khi ảnh thiếu sáng/kính lóa)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        # Tăng độ nhạy (sensitive) để dễ nhận diện hơn (giống như trong app.py)
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # Không tìm thấy khuôn mặt
    if len(faces) == 0:
        return None, img

    # Lấy khuôn mặt đầu tiên và scale tọa độ về ảnh gốc
    x, y, w, h = faces[0]
    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

    # Cắt khuôn mặt từ ảnh gốc (chất lượng cao hơn)
    face_roi = img[y:y + h, x:x + w]

    # Resize về kích thước chuẩn
    face_resized = cv2.resize(face_roi, (64, 64))

    return face_resized, img


if __name__ == "__main__":

    image_path = "anh.jpg"

    face, original = detect_and_crop(image_path)

    if face is not None:
        print("Đã phát hiện khuôn mặt.")
        print("Kích thước:", face.shape)

        cv2.imwrite("face_crop.jpg", face)

        cv2.imshow("Original Image", original)
        cv2.imshow("Face Crop", face)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Không phát hiện được khuôn mặt.")