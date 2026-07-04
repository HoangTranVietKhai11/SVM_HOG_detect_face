import cv2
import numpy as np

"""
Thông số HOG

_WinSize      : Kích thước cửa sổ đầu vào.
                Dùng 64x64 vì khuôn mặt gần vuông.

_BlockSize    : Mỗi block gồm 2x2 cells (16x16 pixel),
                dùng để normalize độ sáng.

_BlockStride  : Block dịch chuyển 8 pixel mỗi lần,
                giúp các block chồng lấp 50%.

_CellSize     : Mỗi cell 8x8 pixel.

_nbins         : 9 hướng gradient (0° - 180°),
                mỗi bin tương ứng 20°.
"""

def create_hog():
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    return hog


# Khởi tạo HOG chỉ một lần
hog = create_hog()


def extract_hog(face_img):
    """
    Trích xuất đặc trưng HOG từ ảnh khuôn mặt.

    Parameters
    ----------
    face_img : numpy.ndarray
        Ảnh BGR đầu vào.

    Returns
    -------
    numpy.ndarray
        Vector HOG dạng 1 chiều.
    """

    if face_img is None:
        raise ValueError("Ảnh đầu vào bị None.")

    # Resize về đúng kích thước nếu cần
    if face_img.shape[:2] != (64, 64):
        face_img = cv2.resize(face_img, (64, 64))

    # Chuyển sang grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Tính đặc trưng HOG
    hog_vector = hog.compute(gray)

    # Trả về vector 1 chiều
    return hog_vector.flatten()


# ===========================
# Test
# ===========================
if __name__ == "__main__":
    # Test với ảnh khuôn mặt đã crop từ face_detector.py
    import sys
    sys.path.append(".")
    from face_detector import detect_and_crop
    
    face, _ = detect_and_crop("face_img.jpg")  # sửa tên ảnh
    
    if face is not None:
        vector = extract_hog(face)
        print("Chiều dài HOG vector:", len(vector))
        print("5 giá trị đầu:", vector[:5])
    else:
        print("Không detect được khuôn mặt")