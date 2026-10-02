import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import winsound

# ----------------------------
# Mediapipe setup
# ----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Eye landmark indexes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ----------------------------
# Functions
# ----------------------------
def eye_aspect_ratio(landmarks, eye_points):
    p1 = np.array([landmarks[eye_points[1]][0], landmarks[eye_points[1]][1]])
    p2 = np.array([landmarks[eye_points[2]][0], landmarks[eye_points[2]][1]])
    p3 = np.array([landmarks[eye_points[4]][0], landmarks[eye_points[4]][1]])
    p4 = np.array([landmarks[eye_points[5]][0], landmarks[eye_points[5]][1]])
    p5 = np.array([landmarks[eye_points[0]][0], landmarks[eye_points[0]][1]])
    p6 = np.array([landmarks[eye_points[3]][0], landmarks[eye_points[3]][1]])
    A = np.linalg.norm(p2 - p4)
    B = np.linalg.norm(p1 - p3)
    C = np.linalg.norm(p5 - p6)
    return (A + B) / (2.0 * C)

# Alarm function
def sound_alarm():
    while alarm_on_global:
        winsound.Beep(2500, 500)  # frequency=2500Hz, duration=500ms
        time.sleep(0.1)

# ----------------------------
# Main program
# ----------------------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("Drowsiness Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Drowsiness Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_counter = 0
CLOSED_FRAMES = 3
EAR_THRESHOLD = 0.40
alarm_on_global = False
alarm_thread = None

# Eyes closed timer
eyes_closed_start = None
eyes_closed_seconds = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    overlay = frame.copy()

    if results.multi_face_landmarks:
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h)))

        leftEAR = eye_aspect_ratio(landmarks, LEFT_EYE)
        rightEAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (leftEAR + rightEAR) / 2.0

        # ----------------------------
        # Interface Background Panels
        # ----------------------------
        cv2.rectangle(overlay, (20, 20), (350, 120), (0, 0, 0), -1)      # EAR box
        cv2.rectangle(overlay, (w - 320, 20), (w - 20, 120), (0, 0, 0), -1)  # Timer box

        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # ----------------------------
        # EAR value
        # ----------------------------
        cv2.putText(frame, f"EAR: {ear:.2f}", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # ----------------------------
        # Eyes Closed Counter
        # ----------------------------
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if eyes_closed_start is None:
                eyes_closed_start = time.time()
            eyes_closed_seconds = int(time.time() - eyes_closed_start)

            if frame_counter >= CLOSED_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (int(w/2) - 300, int(h/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
                if not alarm_on_global:
                    alarm_on_global = True
                    if alarm_thread is None or not alarm_thread.is_alive():
                        alarm_thread = threading.Thread(target=sound_alarm)
                        alarm_thread.start()
        else:
            frame_counter = 0
            alarm_on_global = False
            eyes_closed_start = None
            eyes_closed_seconds = 0

        cv2.putText(frame, f"Eyes Closed: {eyes_closed_seconds}s", (w - 300, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # ----------------------------
    # Show Frame
    # ----------------------------
    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
alarm_on_global = False
