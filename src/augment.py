import cv2
import numpy as np

def augment_real(face):
    """
    Tăng cường dữ liệu cho ảnh thật: Lật ngang, xoay góc ±10 độ và thay đổi độ sáng.
    Được đồng bộ 100% với logic toán học trong Phụ lục B.2 của Báo cáo.
    """
    variants = [face]
    
    # Lật ngang
    variants.append(cv2.flip(face, 1))
    
    # Xoay ±10 độ
    for angle in (-10, 10):
        M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
        variants.append(cv2.warpAffine(face, M, (64, 64)))
        
    # Thay đổi độ sáng
    for beta in (-30, 30):
        variants.append(cv2.convertScaleAbs(face, alpha=1.0, beta=beta))
    
    return variants

def make_synthetic_spoof(face):
    """
    Tạo 2 kiểu dữ liệu giả mạo (spoof): Ảnh in giấy (printed) và ảnh hiển thị qua màn hình (screen replay).
    Đồng bộ 100% với logic code trong Phụ lục B.2 của Báo cáo (dùng circle mask cho vignette).
    """
    # Đảm bảo ảnh ở đúng kích thước 64x64 như giả định trong B.2
    if face.shape[:2] != (64, 64):
        face = cv2.resize(face, (64, 64))

    # 1. Giả lập ảnh in giấy (printed)
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.5                            # giảm bão hoà: mô phỏng ảnh in
    printed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    noise = np.random.normal(0, 8, face.shape)
    printed_with_noise = np.clip(printed.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    printed = cv2.GaussianBlur(printed_with_noise, (3, 3), 0)
    
    # 2. Giả lập màn hình (screen replay) với vignette hình tròn
    mask = cv2.GaussianBlur(cv2.circle(
        np.zeros(face.shape[:2], np.float32), (32, 32), 38, 1.0, -1), (25, 25), 0)
    screen = (face.astype(np.float32) * mask[..., None]).astype(np.uint8)
    
    return [printed, screen]
