import cv2
import time
import base64
import requests
import threading
import asyncio
import edge_tts
import subprocess
import numpy as np
from queue import Queue
from picamera2 import Picamera2

# ───────────────── CONFIG ─────────────────
API_KEY = "AIzaSyCPRlR7YxUqgY4DUscjKG_JhyeHPqCATDY"
MODEL_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={API_KEY}"

FRAME_DELAY = 20
SCENE_THRESHOLD = 30

tts_queue = Queue()

# ───────────────── CAMERA ─────────────────
class Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        ))
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        self.picam2.start()
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        print("CAMERA THREAD STARTED")
        while True:
            try:
                # picamera2 zaten RGB veriyor, dönüşüm yok
                frame = self.picam2.capture_array()
                with self.lock:
                    self.frame = frame
            except Exception as e:
                print("CAMERA FAIL:", e)
                time.sleep(0.5)

    def get(self):
        with self.lock:
            return self.frame


# ───────────────── SCENE CHANGE ─────────────────
def scene_changed(frame1, frame2):
    if frame1 is None or frame2 is None:
        return True
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    score = np.mean(cv2.absdiff(gray1, gray2))
    print(f"SCENE DIFF: {score:.1f}")
    return score > SCENE_THRESHOLD


# ───────────────── AI ─────────────────
def ask_ai(image_b64, retry=3):
    for attempt in range(retry):
        try:
            payload = {
                "contents": [{
                    "parts": [
                        {"text": "Bu görüntüyü Türkçe açıkla kısa ve net."},
                        {"inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64
                        }}
                    ]
                }]
            }

            r = requests.post(MODEL_URL, json=payload, timeout=20)

            if r.status_code == 429:
                print("⚠️ Rate limit! 30s bekleniyor...")
                time.sleep(30)
                continue

            if r.status_code == 503:
                print(f"⚠️ Sunucu yoğun! Deneme {attempt+1}/3, 10s bekleniyor...")
                time.sleep(10)
                continue

            if r.status_code != 200:
                print(f"API ERROR: {r.status_code} - {r.text[:200]}")
                return None

            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:
            print("AI ERROR:", e)
            return None

    print("❌ 3 denemede yanıt alınamadı, atlanıyor.")
    return None


# ───────────────── TTS ─────────────────
async def speak(text):
    try:
        communicate = edge_tts.Communicate(text, "tr-TR-EmelNeural")
        file = "/tmp/tts.mp3"
        await communicate.save(file)
        subprocess.run(["paplay", file])
        subprocess.run(["rm", "-f", file])
    except Exception as e:
        print("TTS ERROR:", e)


def tts_worker():
    while True:
        text = tts_queue.get()
        if text:
            asyncio.run(speak(text))


# ───────────────── MAIN ─────────────────
def main():
    print("SYSTEM STARTING...")

    cam = Camera().start()
    threading.Thread(target=tts_worker, daemon=True).start()

    time.sleep(2)

    last_time = 0
    last_text = ""
    last_frame = None

    print("SYSTEM READY")

    while True:
        frame = cam.get()

        if frame is None:
            time.sleep(0.5)
            continue

        now = time.time()

        if now - last_time < FRAME_DELAY:
            time.sleep(1)
            continue

        if not scene_changed(last_frame, frame):
            print("SCENE SAME, SKIPPING...")
            last_time = now
            last_frame = frame.copy()
            continue

        last_time = now
        last_frame = frame.copy()

        # RGB olarak encode et — Gemini doğru renkleri görür
        small = cv2.resize(frame, (480, 360))
        _, buffer = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        print("CALLING AI...")
        result = ask_ai(img_b64)

        if result:
            print("📢 AI:", result)
            if result != last_text:
                tts_queue.put(result)
                last_text = result


if __name__ == "__main__":
    main()