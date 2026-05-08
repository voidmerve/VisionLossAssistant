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
from collections import deque
from picamera2 import Picamera2
from transformers import pipeline

# ───────────────── CONFIG ─────────────────
API_KEY = "AIzaSyCj_2Uuqgor95crjMxUN2bbPFjIv3tWIcs"
MODEL_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent?key={API_KEY}"

HTTP_TIMEOUT = 15
RATE_LIMIT_DELAY = 2

DEPTH_CALIBRATION = 1.0

# 🔹 BULANIKLIK eşiği (Laplacian variance)
# Yüksek = net, Düşük = bulanık. Pi kamera için 100-150 iyi başlangıç.
BLUR_THRESHOLD = 100

# 🔹 Önceki değerden bu kadar fazla saparsa "outlier" sayılır
DEPTH_JUMP_THRESHOLD = 1.5  # 1.5 metre

# 🔹 Smoothing — son N ölçümün medyanını al
DEPTH_HISTORY_SIZE = 3

tts_queue = Queue(maxsize=1)
tts_busy = threading.Event()

# Depth Anything METRIC modeli
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


# ───────────────── BULANIKLIK TESPİTİ ─────────────────
def is_blurry(frame_rgb):
    """Laplacian variance ile bulanıklığı ölç.
    Yüksek değer = net, Düşük değer = bulanık.
    Bulanıksa True döner."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance, variance < BLUR_THRESHOLD


# ───────────────── BEKLE NETLEŞENE KADAR ─────────────────
def wait_for_sharp_frame(cam, max_wait=2.0):
    """En fazla max_wait saniye boyunca net frame yakalamayı dener.
    Net frame'i veya en az bulanık olanı döner."""
    t0 = time.time()
    best_frame = None
    best_variance = 0

    while time.time() - t0 < max_wait:
        frame = cam.get()
        if frame is None:
            time.sleep(0.05)
            continue

        variance, blurry = is_blurry(frame)

        # Şimdiye kadarkinin en netini sakla
        if variance > best_variance:
            best_variance = variance
            best_frame = frame

        # Net frame bulduk → hemen dön
        if not blurry:
            print(f"  ✓ Net frame yakalandı (blur: {variance:.0f})")
            return frame

        time.sleep(0.1)

    # Süre doldu, en netini ver (hâlâ bulanık olabilir)
    print(f"  ⚠️ Net frame bulunamadı, en iyisi (blur: {best_variance:.0f})")
    return best_frame


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
        top_cut = int(h * 0.30)
        bottom_cut = int(h * 0.65)
        roi = depth_map[top_cut:bottom_cut, :]

        def median_of_closest(region, percent=0.30):
            flat = region.flatten()
            flat = flat[flat > 0.1]
            flat = flat[flat < 15.0]
            if len(flat) == 0:
                return 5.0
            sorted_vals = np.sort(flat)
            cutoff = max(1, int(len(sorted_vals) * percent))
            closest = sorted_vals[:cutoff]
            return float(np.median(closest))

        left = median_of_closest(roi[:, :w//3]) * DEPTH_CALIBRATION
        center = median_of_closest(roi[:, w//3 : 2*w//3]) * DEPTH_CALIBRATION
        right = median_of_closest(roi[:, 2*w//3:]) * DEPTH_CALIBRATION

        return {"left": left, "center": center, "right": right}

    except Exception as e:
        print(f"DEPTH ERROR: {type(e).__name__}: {e}")
        return None


# ───────────────── DEPTH SMOOTHING ─────────────────
class DepthSmoother:
    """Son N ölçümün medyanını alarak gürültüyü filtreler.
    Ani sıçramaları (outlier) reddeder."""

    def __init__(self, history_size=3, jump_threshold=1.5):
        self.history = {
            "left": deque(maxlen=history_size),
            "center": deque(maxlen=history_size),
            "right": deque(maxlen=history_size)
        }
        self.last_smoothed = None
        self.jump_threshold = jump_threshold

    def update(self, depth_info):
        """Yeni ölçümü ekle, smoothed değer döner.
        Eğer ölçüm önceki değerden çok farklıysa REDDET."""
        if not depth_info:
            return self.last_smoothed

        # Önceki değerle karşılaştır — ani sıçrama varsa atla
        if self.last_smoothed is not None:
            for key in ["left", "center", "right"]:
                diff = abs(depth_info[key] - self.last_smoothed[key])
                if diff > self.jump_threshold:
                    print(f"  🚫 Outlier: {key} {self.last_smoothed[key]:.1f} → {depth_info[key]:.1f} (fark: {diff:.1f}m)")
                    return self.last_smoothed  # eski değeri kullan

        # Geçerli ölçüm — geçmişe ekle
        for key in ["left", "center", "right"]:
            self.history[key].append(depth_info[key])

        # Son N ölçümün medyanı
        smoothed = {
            key: float(np.median(list(self.history[key])))
            for key in ["left", "center", "right"]
        }
        self.last_smoothed = smoothed
        return smoothed


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

    parts = [p.strip() for p in spoken.split(",")]
    new_parts = []

    for part in parts:
        part_lower = part.lower()
        added = False
        for direction, dist in direction_map.items():
            if direction in part_lower and dist is not None:
                idx = part_lower.find(direction)
                rest = part[idx:]
                if rest:
                    rest = rest[0].lower() + rest[1:]
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

    smoother = DepthSmoother(
        history_size=DEPTH_HISTORY_SIZE,
        jump_threshold=DEPTH_JUMP_THRESHOLD
    )
    last_text = ""

    while True:
        # 🔹 1. NET FRAME bekle (bulanıksa atla)
        print("→ Net frame aranıyor...")
        frame = wait_for_sharp_frame(cam, max_wait=2.0)

        if frame is None:
            time.sleep(0.5)
            continue

        # Bulanıklık kontrolü (son güvenlik)
        variance, blurry = is_blurry(frame)
        if blurry:
            print(f"  ⚠️ Frame hâlâ bulanık (blur: {variance:.0f}), atlanıyor")
            time.sleep(0.5)
            continue

        # 🔹 2. Frame'i hazırla
        small = cv2.resize(frame, (320, 240))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 50])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        print(f"→ Frame işleniyor (blur: {variance:.0f})...")

        # 🔹 3. AI + Depth paralel
        t_total = time.time()
        ai_text, depth_info_raw = parallel_process(frame, img_b64)
        print(f"  ⏱️ Toplam: {time.time() - t_total:.1f}s")

        # 🔹 4. Depth değerini SMOOTH et (outlier reddet)
        depth_info = smoother.update(depth_info_raw)

        if depth_info_raw:
            print(f"   📏 Ham:    Sol: {depth_info_raw['left']:.2f}m | "
                  f"Orta: {depth_info_raw['center']:.2f}m | "
                  f"Sağ: {depth_info_raw['right']:.2f}m")
        if depth_info:
            print(f"   📏 Smooth: Sol: {depth_info['left']:.2f}m | "
                  f"Orta: {depth_info['center']:.2f}m | "
                  f"Sağ: {depth_info['right']:.2f}m")

        spoken = combine_results(ai_text, depth_info)

        if spoken:
            print(f"📢 {spoken}")

            if spoken.lower() != last_text.lower():
                last_text = spoken
                tts_queue.put(spoken)

                t_start = time.time()
                while not tts_busy.is_set() and time.time() - t_start < 3:
                    time.sleep(0.05)

                while tts_busy.is_set():
                    time.sleep(0.2)

                print(f"⏳ {RATE_LIMIT_DELAY}s bekleniyor...")
                time.sleep(RATE_LIMIT_DELAY)
            else:
                print("⏭️ Aynı sonuç, atlanıyor")
                time.sleep(1)
        else:
            time.sleep(1)


if __name__ == "__main__":
    main()