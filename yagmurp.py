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
MODEL_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent?key={API_KEY}"

HTTP_TIMEOUT = 15

tts_queue = Queue(maxsize=1)
tts_busy = threading.Event()


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
                frame = self.picam2.capture_array()
                with self.lock:
                    self.frame = frame
            except Exception as e:
                print("CAMERA FAIL:", e)
                time.sleep(0.3)

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# ───────────────── AI ─────────────────
def ask_ai(img_b64):
    payload = {
        "contents": [{
            "parts": [
                {"text": "Önündeki en belirgin nesneyi tek kelimeyle Türkçe söyle. Sadece nesne adı, başka hiçbir şey. Örnek: 'masa', 'sandalye', 'insan', 'kapı', 'merdiven'."},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 10,
            "temperature": 0.1
        }
    }

    try:
        t0 = time.time()
        r = requests.post(MODEL_URL, json=payload, timeout=HTTP_TIMEOUT)
        elapsed = time.time() - t0
        print(f"  ⚡ AI: {elapsed:.1f}s - {r.status_code}")

        if r.status_code != 200:
            if r.status_code == 503:
                print("⚠️ 503, atlanıyor")
            elif r.status_code == 429:
                print("⚠️ 429 rate limit, 10s bekle")
                time.sleep(10)
            else:
                print(f"API ERROR: {r.text[:200]}")
            return None

        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            print(f"⚠️ Boş cevap, finishReason: {candidates[0].get('finishReason')}")
            return None

        text = parts[0].get("text", "").strip()
        return text if text else None

    except requests.Timeout:
        print("⏱️ Timeout")
        return None
    except Exception as e:
        print(f"AI ERROR: {type(e).__name__}: {e}")
        return None


# ───────────────── TTS ─────────────────
async def speak(text):
    try:
        communicate = edge_tts.Communicate(text, "tr-TR-EmelNeural")
        file = "/tmp/tts.mp3"
        await communicate.save(file)
        subprocess.run(["paplay", file], check=False)
    except Exception as e:
        print("TTS ERROR:", e)


def tts_worker():
    while True:
        text = tts_queue.get()
        if not text:
            continue
        tts_busy.set()
        try:
            asyncio.run(speak(text))
        finally:
            tts_busy.clear()


# ───────────────── MAIN ─────────────────
def main():
    print("SYSTEM STARTING...")
    cam = Camera().start()
    threading.Thread(target=tts_worker, daemon=True).start()
    time.sleep(2)
    print("SYSTEM READY")

    last_text = ""

    while True:
        # 🔹 1. Frame'i AL ve GÖNDER (tetiklendi)
        frame = cam.get()
        if frame is None:
            time.sleep(0.3)
            continue

        print("→ Frame gönderildi")
        small = cv2.resize(frame, (320, 240))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 50])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        # 🔹 2. AI cevabını bekle
        result = ask_ai(img_b64)

        # 🔹 3. TTS'e gönder
        if result:
            clean = result.replace('"', '').replace("'", "").replace(".", "").strip()
            spoken = f"Önünde {clean} var."
            print(f"📢 {spoken}")

            if spoken.lower() != last_text.lower():
                last_text = spoken
                tts_queue.put(spoken)

                # 🔹 4. TTS başlamasını bekle (kuyruktan al, busy set olsun)
                t_start = time.time()
                while not tts_busy.is_set() and time.time() - t_start < 2:
                    time.sleep(0.05)

                # 🔹 5. TTS BİTENE KADAR BEKLE — başka hiçbir şey yapma
                while tts_busy.is_set():
                    time.sleep(0.1)

                print("✓ TTS bitti, yeni frame için hazır")
            else:
                print("⏭️ Aynı sonuç, tekrar söylenmiyor")
                time.sleep(1)  # aynı sahnedeyse biraz bekle
        else:
            # AI cevap vermediyse kısa bekle, tekrar dene
            time.sleep(1)

        # 🔹 6. Döngü başa dön → yeni frame gönder


if __name__ == "__main__":
    main()