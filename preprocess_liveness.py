"""
Train liveness model (real vs fake).
Cách tiếp cận tối ưu:
- Detect & crop khuôn mặt từ ảnh THẬT
- Tạo fake bằng cách biến đổi trực tiếp trên FACE CROP (không cần detect lại)
- Augment để cân bằng dữ liệu
"""
import cv2
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from face_detector import detect_and_crop
from hog_extractor import extract_hog

REAL_DIR = "dataset_liveness/real"


def make_fake_variants(face):
    """Tạo các biến thể fake từ face crop thật."""
    fakes = []

    # 1. Giả lập ảnh in giấy (printed)
    printed = cv2.convertScaleAbs(face, alpha=0.75, beta=30)
    noise = np.random.normal(0, 12, printed.shape).astype(np.int16)
    printed = np.clip(printed.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    printed = cv2.GaussianBlur(printed, (3, 3), 0.8)
    fakes.append(printed)

    # 2. Giả lập màn hình (screen replay)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 55]
    _, buf = cv2.imencode(".jpg", face, encode_param)
    screen = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    screen = cv2.convertScaleAbs(screen, alpha=1.1, beta=10)
    moire = np.zeros_like(screen)
    moire[::2, :] = 8
    screen = np.clip(screen.astype(np.int16) + moire, 0, 255).astype(np.uint8)
    fakes.append(screen)

    # 3. Giả lập ảnh nghiêng/cong (warped)
    h, w = face.shape[:2]
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    off = w * 0.07
    dst = np.float32([[off, off], [w - off*0.5, off*0.5],
                      [off*0.5, h - off*0.5], [w - off, h - off]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(face, M, (w, h))
    fakes.append(warped)

    return fakes  # 3 biến thể fake


def augment_face(face):
    """Augment ảnh thật: lật + xoay."""
    variants = [face, cv2.flip(face, 1)]
    h, w = face.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        variants.append(cv2.warpAffine(face, M, (w, h)))
    return variants  # 4 biến thể


def load_liveness_dataset():
    X_real, X_fake = [], []

    real_files = [f for f in os.listdir(REAL_DIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Đang detect khuôn mặt trong {len(real_files)} ảnh thật...\n")

    detected = 0
    for filename in real_files:
        img_path = os.path.join(REAL_DIR, filename)
        face, _ = detect_and_crop(img_path)

        if face is None:
            print(f"  Bỏ qua (không detect được mặt): {filename}")
            continue

        detected += 1

        # Ảnh THẬT: gốc + augment
        for real_face in augment_face(face):
            X_real.append(extract_hog(real_face))

        # Ảnh GIẢ: tạo từ face crop
        for fake_face in make_fake_variants(face):
            X_fake.append(extract_hog(fake_face))
            # Augment fake thêm 1 lần lật để tăng số lượng
            X_fake.append(extract_hog(cv2.flip(fake_face, 1)))

    print(f"Detect thành công: {detected}/{len(real_files)} ảnh")
    print(f"Samples real: {len(X_real)} | Samples fake: {len(X_fake)}")

    X = np.array(X_real + X_fake)
    y = np.array(["real"] * len(X_real) + ["fake"] * len(X_fake))
    return X, y


def train_liveness():
    print("=== BẮT ĐẦU TRAIN MODEL LIVENESS ===\n")

    X, y = load_liveness_dataset()

    if len(X) == 0:
        print("Không load được ảnh nào!")
        return

    unique = set(y)
    print(f"\nTổng samples: {len(X)}")
    print(f"Phân bố: { {l: list(y).count(l) for l in unique} }")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n=== TRAIN SVM ===")
    model = LinearSVC(C=1.0, max_iter=3000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== KẾT QUẢ ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_liveness.pkl")
    print("Đã lưu model vào models/model_liveness.pkl")


if __name__ == "__main__":
    train_liveness()
