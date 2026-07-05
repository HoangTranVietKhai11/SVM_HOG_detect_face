"""
Tạo ảnh fake (giả mạo) từ ảnh thật bằng cách mô phỏng:
- Ảnh in giấy (printed photo attack)
- Ảnh hiển thị trên màn hình (screen replay attack)
"""
import cv2
import numpy as np
import os

REAL_DIR = "dataset_liveness/real"
FAKE_DIR = "dataset_liveness/fake"


def simulate_printed_photo(img):
    """Mô phỏng ảnh bị in ra giấy rồi chụp lại."""
    # Giảm độ tương phản + fade màu
    result = cv2.convertScaleAbs(img, alpha=0.75, beta=30)

    # Thêm noise hạt (giấy in)
    noise = np.random.normal(0, 12, result.shape).astype(np.int16)
    result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Blur nhẹ (mất nét khi in)
    result = cv2.GaussianBlur(result, (3, 3), 0.8)

    # Giảm saturation (màu nhạt hơn)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.6
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


def simulate_screen_replay(img):
    """Mô phỏng ảnh hiển thị trên màn hình điện thoại/laptop."""
    # JPEG compression artifact (màn hình có pixel)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 55]
    _, buffer = cv2.imencode(".jpg", img, encode_param)
    result = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # Thêm vignette (viền tối quanh màn hình)
    rows, cols = result.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols * 0.6)
    kernel_y = cv2.getGaussianKernel(rows, rows * 0.6)
    vignette = kernel_y * kernel_x.T
    vignette = vignette / vignette.max()
    result = (result * vignette[:, :, np.newaxis]).astype(np.uint8)

    # Tăng nhẹ độ sáng (ánh sáng màn hình)
    result = cv2.convertScaleAbs(result, alpha=1.1, beta=10)

    # Moiré pattern nhẹ (pixel grid)
    moire = np.zeros_like(result)
    moire[::2, :] = 8
    result = np.clip(result.astype(np.int16) + moire, 0, 255).astype(np.uint8)

    return result


def simulate_warped_photo(img):
    """Mô phỏng ảnh bị cầm nghiêng/cong."""
    rows, cols = img.shape[:2]

    # Warp perspective nhẹ
    src = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    offset = cols * 0.05
    dst = np.float32([
        [offset, offset],
        [cols - offset * 0.5, offset * 0.5],
        [offset * 0.5, rows - offset * 0.5],
        [cols - offset, rows - offset]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(img, M, (cols, rows))

    # Thêm shadow nhẹ (bóng khi cầm)
    shadow = np.ones_like(result, dtype=np.float32)
    shadow[:, :cols // 3] = 0.85
    result = (result * shadow).astype(np.uint8)

    return result


def generate_fake_images():
    os.makedirs(FAKE_DIR, exist_ok=True)

    real_files = [f for f in os.listdir(REAL_DIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"=== TẠO ẢNH FAKE TỪ {len(real_files)} ẢNH THẬT ===\n")

    count = 0
    for filename in real_files:
        img_path = os.path.join(REAL_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  Bỏ qua (không đọc được): {filename}")
            continue

        name, ext = os.path.splitext(filename)

        # Tạo 3 biến thể fake cho mỗi ảnh thật
        variants = {
            "printed": simulate_printed_photo(img),
            "screen":  simulate_screen_replay(img),
            "warped":  simulate_warped_photo(img),
        }

        for variant_name, fake_img in variants.items():
            out_path = os.path.join(FAKE_DIR, f"fake_{variant_name}_{name}.jpg")
            cv2.imwrite(out_path, fake_img)
            count += 1

        print(f"  ✓ {filename} → 3 ảnh fake")

    print(f"\nTổng ảnh fake đã tạo: {count}")
    print(f"Lưu tại: {FAKE_DIR}/")


if __name__ == "__main__":
    generate_fake_images()
