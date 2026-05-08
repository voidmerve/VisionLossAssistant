import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import time
import base64
import requests
import threading
import asyncio
import edge_tts
import subprocess
import numpy as np
import torch
from PIL import Image
from queue import Queue
from picamera2 import Picamera2
from transformers import pipeline

# ───────────────── CONFIG ─────────────────
API_KEY = "AIzaSyAP0kqRpSKvbPf8t7qFg8L1T-qWJGvpSIc"
MODEL_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent?key={API_KEY}"

HTTP_TIMEOUT = 15
RATE_LIMIT_DELAY = 2  # TTS bittikten sonra ek bekleme (saniye)

tts_queue = Queue(maxsize=1)
tts_busy = threading.Event()

# Depth Anything METRIC modeli (gerçek metre döner)
print("DEPTH MODELİ YÜKLENİYOR...")
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    device=-1
)
print("DEPTH MODELİ HAZIR")


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


# ───────────────── DEPTH ESTIMATION ─────────────────
def estimate_depth(frame_rgb):
    """Sol/orta/sağ bölgelerin metre cinsinden mesafesini döner."""
    try:
        t0 = time.time()
        small = cv2.resize(frame_rgb, (256, 192))
        pil_img = Image.fromarray(small)

        result = depth_pipe(pil_img)
        if "predicted_depth" in result:
            depth_map = np.array(result["predicted_depth"])
        else:
            depth_map = np.array(result["depth"])

        elapsed = time.time() - t0
        print(f"  🔍 Depth: {elapsed:.1f}s")

        h, w = depth_map.shape
        roi = depth_map[h//4 : 3*h//4, :]

        # En yakın %25'in ortalaması = daha gerçekçi
        def closest_quarter_mean(region):
            flat = region.flatten()
            sorted_vals = np.sort(flat)
            return float(np.mean(sorted_vals[:max(1, len(sorted_vals)//4)]))

        left = closest_quarter_mean(roi[:, :w//3])
        center = closest_quarter_mean(roi[:, w//3 : 2*w//3])
        right = closest_quarter_mean(roi[:, 2*w//3:])

        return {"left": left, "center": center, "right": right}

    except Exception as e:
        print(f"DEPTH ERROR: {type(e).__name__}: {e}")
        return None


# ───────────────── AI (GEMINI) ─────────────────
def ask_ai(img_b64):
    payload = {
        "contents": [{
            "parts": [
                {"text": (
                    "Görüntüde gördüğün en fazla 3 nesneyi yön bilgisiyle birlikte tek cümlede Türkçe söyle. "
                    "Format: 'Önünde X var, sağında Y, solunda Z.' "
                    "Eğer 2 nesne varsa: 'Önünde X var, sağında Y.' veya 'Sağında X var, solunda Y.' gibi. "
                    "Eğer 1 nesne varsa: 'Önünde X var.' "
                    "Yön: nesne görüntünün ortasındaysa 'önünde', sağ tarafındaysa 'sağında', sol tarafındaysa 'solunda'. "
                    "Sadece bu format, başka hiçbir şey yazma."
                )},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 60,
            "temperature": 0.1
        }
    }

    try:
        t0 = time.time()
        r = requests.post(MODEL_URL, json=payload, timeout=HTTP_TIMEOUT)
        elapsed = time.time() - t0
        print(f"  ⚡ AI: {elapsed:.1f}s - {r.status_code}")

        if r.status_code == 429:
            print("  ⚠️ Rate limit, biraz bekleniyor...")
            time.sleep(5)
            return None

        if r.status_code != 200:
            return None

        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return None

        return parts[0].get("text", "").strip()

    except requests.Timeout:
        print("⏱️ Timeout")
        return None
    except Exception as e:
        print(f"AI ERROR: {type(e).__name__}: {e}")
        return None


# ───────────────── PARALEL İŞLEM ─────────────────
def parallel_process(frame_rgb, img_b64):
    results = {"ai": None, "depth": None}

    def run_ai():
        results["ai"] = ask_ai(img_b64)

    def run_depth():
        results["depth"] = estimate_depth(frame_rgb)

    t1 = threading.Thread(target=run_ai)
    t2 = threading.Thread(target=run_depth)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return results["ai"], results["depth"]


# ───────────────── ÇIKTI BİRLEŞTİRME ─────────────────
def combine_results(ai_text, depth_info):
    """
    Gemini cevabındaki her yön kelimesine o yönün mesafesini ekler.
    
    Örnekler:
        "Önünde insan var."
        → "1.8 önünde insan var"
        
        "Önünde masa var, sağında sandalye."
        → "1.3 önünde masa var, 1.6 sağında sandalye"
    """
    if not ai_text:
        return None

    spoken = ai_text.replace('"', '').replace("'", "").strip()

    if not depth_info:
        return spoken

    direction_map = {
        "önünde": depth_info["center"],
        "sağında": depth_info["right"],
        "solunda": depth_info["left"],
    }

    # Cümleyi virgülle parçalara ayır
    parts = [p.strip() for p in spoken.split(",")]
    new_parts = []

    for part in parts:
        part_lower = part.lower()
        added = False
        for direction, dist in direction_map.items():
            if direction in part_lower and dist is not None:
                idx = part_lower.find(direction)
                rest = part[idx:]
                # İlk harfi küçült
                if rest:
                    rest = rest[0].lower() + rest[1:]
                # Sondaki noktayı temizle
                rest = rest.rstrip(".")
                new_parts.append(f"{dist:.1f} {rest}")
                added = True
                break

        if not added:
            new_parts.append(part.rstrip("."))

    return ", ".join(new_parts)


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
        frame = cam.get()
        if frame is None:
            time.sleep(0.3)
            continue

        small = cv2.resize(frame, (320, 240))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 50])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        print("→ Frame işleniyor (paralel)...")
        t_total = time.time()
        ai_text, depth_info = parallel_process(frame, img_b64)
        print(f"  ⏱️ Toplam: {time.time() - t_total:.1f}s")

        if depth_info:
            print(f"   📏 Sol: {depth_info['left']:.2f}m | "
                  f"Orta: {depth_info['center']:.2f}m | "
                  f"Sağ: {depth_info['right']:.2f}m")

        spoken = combine_results(ai_text, depth_info)

        if spoken:
            print(f"📢 {spoken}")

            if spoken.lower() != last_text.lower():
                last_text = spoken
                tts_queue.put(spoken)

                # TTS başlamasını bekle
                t_start = time.time()
                while not tts_busy.is_set() and time.time() - t_start < 3:
                    time.sleep(0.05)

                # TTS bitene kadar bekle
                while tts_busy.is_set():
                    time.sleep(0.2)

                # Rate limit'ten kaçınmak için ek bekleme
                print(f"⏳ {RATE_LIMIT_DELAY}s bekleniyor...")
                time.sleep(RATE_LIMIT_DELAY)
            else:
                print("⏭️ Aynı sonuç, atlanıyor")
                time.sleep(1)
        else:
            time.sleep(1)


if __name__ == "__main__":
    main()