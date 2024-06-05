import cv2
import numpy as np

# Mở kết nối với camera (đổi số 0 thành số camera thực tế nếu có nhiều camera)
#cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('C:\\Users\\ASUS\\OneDrive - UET\Documents\\Line_Detection\\cam05.mp4')
while cap.isOpened():
    # Đọc từng khung hình
    ret, frame = cap.read()

    if not ret:
        break

    # Chuyển đổi khung hình sang không gian màu HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Xác định khoảng màu đen trong không gian màu HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])

    # Tạo mask để lọc các pixel màu đen
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Tìm contour của các vùng màu đen trong mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    # Lấy kích thước của khung hình
    height, width, _ = frame.shape

    # Vẽ đường ngang nằm ở 1/5 phía trên màn hình
    horizontal_line = int(1 / 5 * height)
    cv2.line(frame, (0, horizontal_line), (width, horizontal_line), (0, 0, 255), 2)

    # Vẽ đường thẳng đứng ở chính giữa màn hình
    cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 0, 255), 2)

    max_area = 0
    left_area = 0
    right_area = 0
    max_contour = None

    # Tìm đường bao có diện tích lớn nhất và tính diện tích phía bên trái và phía bên phải của đường thẳng đứng
    for contour in contours:
        # Tính diện tích của contour
        area = cv2.contourArea(contour)

        # Xấp xỉ contour bằng đa giác đơn giản
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Lấy tâm của contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Kiểm tra xem tâm của contour có nằm phía dưới đường ngang không
        if cY > horizontal_line:
            # Chỉ xét các contour có diện tích lớn hơn 3000
            if area > 3000:
                # Tìm contour có diện tích lớn nhất
                if area > max_area:
                    max_area = area
                    max_contour = contour

    # Vẽ đường bao của contour có diện tích lớn nhất
    if max_contour is not None:
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

        # Hiển thị diện tích và thông báo rẽ trái, rẽ phải, hoặc đi thẳng
        cv2.putText(frame, f'Max Area: {int(max_area)}', (width // 2 + 10, horizontal_line - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'Left Area: {int(left_area)}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'Right Area: {int(right_area)}', (width // 2 + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Kiểm tra và in thông báo rẽ trái, rẽ phải, hoặc đi thẳng
        if right_area == 0 or left_area == 0:
            direction = "Forward"
        elif left_area > right_area:
            direction = "Turn Left"
        elif right_area > left_area:
            direction = "Turn Right"
        else:
            direction = "Forward"

        #print(f'Load: {direction}')

        # Hiển thị thông báo lên màn hình
        cv2.putText(frame, f'Load: {direction}', (width // 2, horizontal_line + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (133,120, 10), 5)

    # Hiển thị video với đường bao và các đường thẳng
    cv2.imshow('Contours and Lines', frame)

    # Nhấn 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ video
cap.release()
cv2.destroyAllWindows()
