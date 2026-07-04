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

    # Chuyển sang ảnh xám để tăng tốc độ detect
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,

        # Mỗi lần thu nhỏ ảnh 10% để tạo Image Pyramid.
        # Giá trị càng gần 1 thì detect càng kỹ nhưng sẽ chậm hơn.
        scaleFactor=1.1,

        # Một vùng phải được phát hiện ít nhất 5 lần
        # mới được coi là khuôn mặt thật.
        # Giá trị càng lớn thì càng giảm false positive.
        minNeighbors=5,

        # Bỏ qua những khuôn mặt nhỏ hơn 48x48 pixel.
        # Giúp tránh nhận nhầm các vật thể nhỏ.
        minSize=(48, 48)
    )

    # Không tìm thấy khuôn mặt
    if len(faces) == 0:
        return None, img

    # Lấy khuôn mặt đầu tiên
    x, y, w, h = faces[0]

    # Cắt khuôn mặt
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