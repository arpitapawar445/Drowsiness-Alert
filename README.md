# Drowsiness-Alert 😴🚨

A real-time driver drowsiness detector. It watches your eyes through the webcam and
rings an alarm (sound + voice) plus sends an SMS when your eyes stay closed too long.

## How it works (simple)

- The camera finds your face and measures your eyes 30 times per second.
- Open eyes give a big number (called EAR, e.g. `0.52`). Closed eyes give a small number (e.g. `0.20`).
- You set a line in the middle (called threshold, e.g. `0.40`).
- If your number stays **below the line** for a few seconds → **DROWSINESS ALERT!** 🔔📩

## Calibrate button 📏 (easy words)

Every person's eyes look different on camera, so one fixed line does not fit all.
Pressing **Calibrate** with your eyes **open** does this:

1. It looks at your open eyes for 3 seconds and learns your normal number.
2. It draws the line a little below that (about 3/4 of your number).
3. Now closing your eyes crosses the line and the alarm rings by itself.

Example: your open number is `0.52` → Calibrate sets the line to `0.40` →
closing your eyes (`0.20`) crosses it → alarm + SMS automatically. No clicking needed.
Press Calibrate again if light or seating distance changes a lot.

## Two ways to run

### 1. Browser demo (easiest, no install) 🌐

- Just open `index.html`, or use the live link: `https://arpitapawar445.github.io/Drowsiness-Alert/`
- Runs 100% in the browser. Camera never leaves your device.
- Gives sound + voice alarm. **No SMS** here (a web page alone cannot send texts).

### 2. Full version with SMS (on your computer) 📩

Needs Python + your free Twilio keys. Steps:

```bash
git clone https://github.com/arpitapawar445/Drowsiness-Alert.git
cd Drowsiness-Alert
python -m venv venv
venv\Scripts\activate        # Windows
pip install flask flask-socketio opencv-python mediapipe numpy twilio python-dotenv
```

Copy `.env.example` to `.env` and fill your values (see below), then:

```bash
python app.py
```

Open `http://127.0.0.1:5000`, press **Start Camera**, then **Calibrate**.
Closing your eyes now rings the alarm **and** texts your phone (max 1 SMS per minute).

## Twilio SMS setup (free trial)

1. Sign up at `twilio.com/try-twilio`, verify email + mobile.
2. From `console.twilio.com` copy **Account SID** + **Auth Token**.
3. Phone Numbers → get a trial number (SMS-capable, e.g. `+1...`) → this is `TWILIO_FROM_NUMBER`.
4. Phone Numbers → Verified Caller IDs → add your mobile (`+91...`) → this is `DROWSINESS_ALERT_PHONE`.
5. Put all four in `.env` (never commit this file):

```
TWILIO_ACCOUNT_SID=ACxxxx...
TWILIO_AUTH_TOKEN=xxxx...
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
DROWSINESS_ALERT_PHONE=+91xxxxxxxxxx
```

Trial accounts only text verified numbers. Check **Monitor → Logs → Messaging** in Twilio if a message doesn't arrive.

## Files

| File | What it is |
|---|---|
| `index.html` | The web page (camera, detection, alarm). Works alone or served by Flask. |
| `app.py` | Flask server: serves the page + `/api/drowsy-alert` endpoint that sends the SMS. |
| `face_landmarker.task` | Face-detection model (loaded in browser, with CDN fallback). |
| `.env.example` | Template for your Twilio keys. Copy to `.env`. |
| `drowsy_basic.py`, `test.py`, `test_cam.py` | Small test/experiment scripts. |

## Controls on the page

- **▶ Start / ⏹ Stop Camera** — start/stop detection.
- **📏 Calibrate** — auto-set threshold from your open eyes (recommended).
- **EAR threshold slider** — manual line (0.15–0.60). Lower = less sensitive.
- **Alert after slider** — how long eyes must stay closed before alarm (1–4 s).
- **🔊 Test sound** — checks speaker/permission only. Real alarm is automatic.
- **Sound + voice checkbox** — mute/unmute alarm.

---

© 2025 Arpita Pawar. Built with Flask, MediaPipe & Twilio. ❤️
