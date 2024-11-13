import cv2
import numpy as np

cap = cv2.VideoCapture("object_video_2.mp4")

if not cap.isOpened():
    print("Failed to open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 180])
    upper = np.array([180, 65, 255])

    mask = cv2.inRange(hsv_frame, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 250:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Detect Object", frame)

    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
