import cv2
import numpy as np

def process_image(image):
    # Chuyển đổi ảnh sang đen trắng
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tạo ngưỡng để tách các vùng đen
    _, blackline = cv2.threshold(grayscale_image, 60, 255, cv2.THRESH_BINARY)

    # Làm mịn và mở rộng để loại bỏ nhiễu
    kernel = np.ones((3, 3), np.uint8)
    blackline = cv2.erode(blackline, kernel, iterations=5)
    blackline = cv2.dilate(blackline, kernel, iterations=9)

    # Tìm các đường viền trong hình ảnh đã xử lý
    contours, _ = cv2.findContours(blackline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Lấy hình chữ nhật bao quanh vùng đen
        blackbox = cv2.minAreaRect(contours[0])
        (x_min, y_min), (w_min, h_min), ang = blackbox
        
        # Xử lý góc quay
        if ang < -45:
            ang = 90 + ang
        if w_min < h_min and ang > 0:
            ang = (90 - ang) * -1
        if w_min > h_min and ang < 0:
            ang = 90 + ang
        
        # Vẽ hình chữ nhật và các đường trên ảnh gốc
        box = cv2.boxPoints(blackbox)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
        
        # Tính lỗi so với điểm setpoint
        setpoint = image.shape[1] // 2
        error = int(x_min - setpoint)
        
        # Hiển thị góc quay và lỗi trên ảnh
        cv2.putText(image, str(int(ang)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(error), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(image, (int(x_min), 200), (int(x_min), 250), (255, 0, 0), 3)

    return image

# Khởi tạo video từ webcam hoặc video từ tệp
cap = cv2.VideoCapture(1)  # 0 là index của webcam (nếu có)
# cap = cv2.VideoCapture('path_to_video_file.mp4')  # Đường dẫn đến tệp video

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_image(frame)
    
    cv2.imshow('Original with line', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng và đóng cửa sổ khi kết thúc
cap.release()
cv2.destroyAllWindows()
