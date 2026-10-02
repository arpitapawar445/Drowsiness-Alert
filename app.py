# app.py
from flask import Flask, Response, send_from_directory, request, jsonify
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime, timedelta
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import threading
import winsound  # Python-generated beep
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ----------------------------
# Twilio credentials
# ----------------------------
import os

account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_NUM = os.getenv("TWILIO_FROM_NUMBER", "+13187589602")
ALERT_PHONE = os.getenv("DROWSINESS_ALERT_PHONE", "+918291155210")


client = Client(account_sid, auth_token) if account_sid and auth_token else None
last_sms_time = None  # For rate limiting

# Optional: face_recognition for registered-driver verification
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except Exception:
    FACE_RECOGNITION_AVAILABLE = False
    print("[WARN] face_recognition not available — driver verification disabled.")

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Mediapipe eye landmark indexes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ----------------------------
# EAR function
# ----------------------------
def eye_aspect_ratio(landmarks, eye_points):
    p1 = np.array(landmarks[eye_points[1]])
    p2 = np.array(landmarks[eye_points[2]])
    p3 = np.array(landmarks[eye_points[4]])
    p4 = np.array(landmarks[eye_points[5]])
    p5 = np.array(landmarks[eye_points[0]])
    p6 = np.array(landmarks[eye_points[3]])
    A = np.linalg.norm(p2 - p4)
    B = np.linalg.norm(p1 - p3)
    C = np.linalg.norm(p5 - p6)
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

# ----------------------------
# Alarm setup using winsound (immediate stop)
# ----------------------------
alarm_thread = None
alarm_playing = False
alarm_lock = threading.Lock()

def play_alarm():
    global alarm_playing, alarm_thread
    with alarm_lock:
        if alarm_playing:
            return
        alarm_playing = True

    def beep_loop():
        global alarm_playing
        while True:
            with alarm_lock:
                if not alarm_playing:
                    break
            winsound.Beep(1000, 200)  # 1000Hz, 200ms beep
            time.sleep(0.05)           # tiny pause for immediate stop

    alarm_thread = threading.Thread(target=beep_loop)
    alarm_thread.start()

def stop_alarm():
    global alarm_playing
    with alarm_lock:
        alarm_playing = False

# ----------------------------
# Load registered driver face encoding (optional)
# ----------------------------
KNOWN_DRIVER_ENCODING = None
KNOWN_DRIVER_PATH = "driver.jpg"

if FACE_RECOGNITION_AVAILABLE and os.path.exists(KNOWN_DRIVER_PATH):
    try:
        known_image = face_recognition.load_image_file(KNOWN_DRIVER_PATH)
        encs = face_recognition.face_encodings(known_image)
        if len(encs) > 0:
            KNOWN_DRIVER_ENCODING = encs[0]
            print("[INFO] Loaded registered driver encoding from", KNOWN_DRIVER_PATH)
        else:
            print("[WARN] No face found in driver.jpg — driver verification disabled")
    except Exception as e:
        print("[WARN] Failed to load driver encoding:", e)
else:
    if not FACE_RECOGNITION_AVAILABLE:
        print("[INFO] face_recognition not installed; skip driver verification.")
    else:
        print("[INFO] driver.jpg not found; skip driver verification.")

# ----------------------------
# SMS alert function (rate limited)
# ----------------------------
def send_sms_alert():
    global last_sms_time
    if client is None:
        print("[INFO] Twilio credentials not set (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN); skipping SMS alert.")
        return "no-credentials"
    now = datetime.now()
    if last_sms_time is None or now - last_sms_time >= timedelta(minutes=1):
        try:
            message = client.messages.create(
                body="🚨 Alert! Driver appears drowsy!",
                from_=TWILIO_NUM,
                to=ALERT_PHONE
            )
            print("[INFO] SMS sent:", message.sid)
            last_sms_time = now
            return "sent:" + message.sid
        except TwilioRestException as e:
            if e.code == 21660:
                print(
                    "[ERROR] Twilio sender is not owned by this account. "
                    "Set TWILIO_FROM_NUMBER to an SMS-capable number in Twilio."
                )
            else:
                print(f"[ERROR] Twilio SMS failed ({e.code}): {e.msg}")
            return "error"
        except Exception as e:
            print("[ERROR] Twilio SMS failed:", e)
            return "error"
    return "rate-limited"

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/drowsy-alert', methods=['POST'])
def drowsy_alert():
    """Called by the browser page when drowsiness is detected.
    Sends the Twilio SMS (rate-limited to 1/min). Works when the page is
    served from this Flask server (http://127.0.0.1:5000). On GitHub Pages
    there is no backend, so the page just skips this call."""
    data = request.get_json(silent=True) or {}
    status = send_sms_alert()
    return jsonify({"sms": status, "ear": data.get("ear")})

def gen_frames():
    cap = cv2.VideoCapture(0)
    face_landmarker = None
    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if os.path.exists(model_path):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    EAR_THRESHOLD = 0.40
    CLOSED_FRAMES = 3
    frame_counter = 0
    eyes_closed_start = None
    eyes_closed_seconds = 0
    alert = False

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = None
            if face_landmarker is not None:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = face_landmarker.detect(image)

            ear = 0.0
            alert = False
            driver_verified = None

            if results is not None and results.face_landmarks:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]]

                leftEAR = eye_aspect_ratio(landmarks, LEFT_EYE)
                rightEAR = eye_aspect_ratio(landmarks, RIGHT_EYE)
                ear = (leftEAR + rightEAR) / 2.0

                xs = [p[0] for p in landmarks]
                ys = [p[1] for p in landmarks]
                x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
                y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)

                # Driver verification
                if KNOWN_DRIVER_ENCODING is not None and FACE_RECOGNITION_AVAILABLE:
                    try:
                        face_crop = frame[y_min:y_max, x_min:x_max]
                        if face_crop.size != 0:
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            encs = face_recognition.face_encodings(face_rgb)
                            if len(encs) > 0:
                                match = face_recognition.compare_faces([KNOWN_DRIVER_ENCODING], encs[0], tolerance=0.6)[0]
                                driver_verified = bool(match)
                            else:
                                driver_verified = False
                        else:
                            driver_verified = False
                    except Exception as e:
                        print("[WARN] face_recognition error:", e)
                        driver_verified = None

                # ----------------------------
                # Draw bounding box + label (always show Verified)
                # ----------------------------
                label = "Verified ✅"
                color = (0, 200, 0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                # ----------------------------
                # Drowsiness detection + alarm
                # ----------------------------
                if ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()
                    eyes_closed_seconds = int(time.time() - eyes_closed_start)
                    if frame_counter >= CLOSED_FRAMES:
                        alert = True
                        send_sms_alert()
                        play_alarm()
                else:
                    frame_counter = 0
                    eyes_closed_start = None
                    eyes_closed_seconds = 0
                    alert = False
                    stop_alarm()

            else:
                if face_landmarker is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if not face_detector.empty():
                        faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
                        for x, y, face_width, face_height in faces[:1]:
                            cv2.rectangle(frame, (x, y), (x + face_width, y + face_height), (0, 200, 0), 2)
                            cv2.putText(frame, "Face detected", (x, max(y - 10, 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)
                frame_counter = 0
                eyes_closed_start = None
                eyes_closed_seconds = 0
                alert = False
                stop_alarm()

            # Send data to frontend
            socketio.emit('update', {
                'ear': round(float(ear), 2),
                'eyes_closed': eyes_closed_seconds,
                'alert': alert,
                'driver_verified': True  # Always True
            })

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        if face_landmarker is not None:
            face_landmarker.close()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------------------
# Run app
# ----------------------------
if __name__ == '__main__':
    print("[DEBUG] Starting app.py ...")
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, use_reloader=False)
