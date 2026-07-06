import cv2
import numpy as np

def augment_real(face):
    """
    Tăng cường dữ liệu cho ảnh thật: Lật ngang, xoay góc ±10 độ và thay đổi độ sáng.
    Trọng tâm khớp với mô tả trong báo cáo (flip, rotate, brightness).
    """
    variants = [face]
    
    # Lật ngang
    variants.append(cv2.flip(face, 1))
    
    h, w = face.shape[:2]
    center = (w // 2, h // 2)
    
    # Xoay ±10 độ
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        variants.append(cv2.warpAffine(face, M, (w, h)))
        
    # Thay đổi độ sáng
    variants.append(cv2.convertScaleAbs(face, alpha=1.2, beta=20))
    variants.append(cv2.convertScaleAbs(face, alpha=0.8, beta=-20))
    
    return variants

def make_synthetic_spoof(face):
    """
    Tạo 2 kiểu dữ liệu giả mạo (spoof): Ảnh in giấy (printed) và ảnh hiển thị qua màn hình (screen replay).
    Khớp chính xác với báo cáo (chỉ 2 loại, screen dùng vignette hình tròn).
    """
    fakes = []

    # 1. Giả lập ảnh in giấy (printed)
    printed = cv2.convertScaleAbs(face, alpha=0.75, beta=30)
    noise = np.random.normal(0, 12, printed.shape).astype(np.int16)
    printed = np.clip(printed.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    printed = cv2.GaussianBlur(printed, (3, 3), 0.8)
    
    # Giảm bão hòa màu cho ảnh in
    hsv = cv2.cvtColor(printed, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.6
    printed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    fakes.append(printed)

    # 2. Giả lập màn hình (screen replay) với vignette hình tròn (khớp báo cáo)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 55]
    _, buf = cv2.imencode(".jpg", face, encode_param)
    screen = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    
    rows, cols = screen.shape[:2]
    # Tạo vignette hình tròn
    X, Y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    d = np.sqrt(X**2 + Y**2)
    vignette = np.clip(1 - (d * 0.8), 0, 1)
    
    screen = (screen * vignette[:, :, np.newaxis]).astype(np.uint8)
    # Tăng độ sáng chói của màn hình
    screen = cv2.convertScaleAbs(screen, alpha=1.1, beta=10)
    
    fakes.append(screen)

    return fakes
