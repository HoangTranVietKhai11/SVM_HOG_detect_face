import cv2
import os
import time

def main():
    print("=== CÔNG CỤ TỰ ĐỘNG THU THẬP ẢNH ===")
    print("Vui lòng chọn người bạn muốn thu thập:")
    print("1. person_a (Khải)")
    print("2. person_b (Đức)")
    print("3. unknown (Người lạ)")
    
    choice = input("Nhập số (1/2/3): ").strip()
    if choice == '1':
        name = "person_a"
    elif choice == '2':
        name = "person_b"
    elif choice == '3':
        name = "unknown"
    else:
        print("Lựa chọn không hợp lệ!")
        return

    # Đường dẫn lưu ảnh gốc (khớp với báo cáo)
    save_dir = f"data/raw/{name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Đếm số ảnh hiện có để không bị ghi đè file cũ
    existing_files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))]
    count = len(existing_files)
    
    print(f"\nĐang lưu vào thư mục: {save_dir}")
    print(f"Đã có sẵn {count} ảnh trong thư mục này.")
    print("====================================")
    print("HƯỚNG DẪN:")
    print("- Nhấn phím 's' để chụp 1 tấm.")
    print("- Nhấn phím 'a' để TỰ ĐỘNG CHỤP LIÊN TỤC (tự động xoay mặt đi nhé).")
    print("- Nhấn phím 'q' để thoát.")
    
    cap = cv2.VideoCapture(0)
    auto_mode = False
    last_capture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể kết nối Camera!")
            break
            
        frame = cv2.flip(frame, 1) # Lật gương cho dễ nhìn
        
        # Chụp ảnh trước khi vẽ hướng dẫn lên màn hình
        clean_frame = frame.copy()
        
        # Vẽ hướng dẫn lên màn hình (chỉ hiển thị)
        cv2.putText(frame, f"Nhan 's': Chup 1 tam | 'a': Tu dong chup | 'q': Thoat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Da chup: {count} anh", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if auto_mode:
            cv2.putText(frame, "AUTO MODE ON", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
        cv2.imshow("Capture Dataset", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a'):
            auto_mode = not auto_mode # Bật/tắt chế độ chụp tự động
            
        # Nếu đang ở chế độ auto và cách lần chụp trước 0.3s -> Chụp!
        if (key == ord('s')) or (auto_mode and time.time() - last_capture_time > 0.3):
            count += 1
            img_path = os.path.join(save_dir, f"img_{int(time.time())}.jpg")
            
            # Lưu ảnh sạch (không có chữ)
            cv2.imwrite(img_path, clean_frame)
            print(f"Đã lưu: {img_path}")
            last_capture_time = time.time()
            
            if auto_mode and count >= 100:
                print("Đã đạt mốc 100 ảnh. Tự động dừng auto!")
                auto_mode = False

    cap.release()
    cv2.destroyAllWindows()
    print("Hoàn thành!")

if __name__ == "__main__":
    main()
