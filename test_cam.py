import cv2

cap = cv2.VideoCapture(0)  # 0 = default laptop camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Webcam Test", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
