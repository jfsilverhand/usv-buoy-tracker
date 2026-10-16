#!/usr/bin/env python3
"""
USV Otonom Hedef Takip Sistemi
GStreamer RTSP + TensorRT YOLOv8 + SIYI Gimbal
"""
import sys
import cv2
import numpy as np
import time
import signal
import threading

from siyi_gimbal import SIYIGimbalController
from pid_controller import PIDController

import os as _os
RTSP_URI    = _os.environ.get("RTSP_URI",  "rtsp://192.168.2.25:8554/main.264")
GIMBAL_IP   = _os.environ.get("GIMBAL_IP", "192.168.2.25")
GIMBAL_PORT = int(_os.environ.get("GIMBAL_PORT", "37256"))

# Host fallback (DeepStream olmadan): OpenCV RTSP + TensorRT/PyCUDA.
# Engine'ler cihaza özeldir; host'ta üret:
#   /usr/src/tensorrt/bin/trtexec --onnx=models/best.onnx --saveEngine=models/best_fp16_host.engine --fp16
_base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_in_docker     = _os.path.exists("/.dockerenv")
_host_engine   = _os.path.join(_base, "models", "best_fp16_host.engine")
_docker_engine = _os.path.join(_base, "models", "best_fp16.engine")
_coco_engine   = _os.path.join(_base, "models",
                               "yolov8n_coco_docker_fp16.engine" if _in_docker else "yolov8n_coco_fp16.engine")
_coco_pt       = _os.path.join(_base, "src", "yolov8n.pt")
ENGINE_PATH = _host_engine if _os.path.exists(_host_engine) else _docker_engine

# ─── DETECTOR_MODE ──────────────────────────────────────────────
# 'trt'      → TRT engine, duba tespiti (models/best_fp16*.engine)
# 'trt_coco' → TRT engine, COCO modeli (su sisesi demo)
# 'coco'     → Ultralytics CPU, COCO (yavaş; 'ultralytics' paketi gerekir, Docker image'da YOK)
if _os.path.exists(ENGINE_PATH):
    DETECTOR_MODE = 'trt'
elif _os.path.exists(_coco_engine):
    DETECTOR_MODE = 'trt_coco'
else:
    DETECTOR_MODE = 'coco'

# COCO modunda hangi class'ları takip et (None = hepsi)
# 39 = bottle, 41 = cup, 0 = person
COCO_TARGET_CLASSES = [0, 39, 41]   # 0=person(test), 39=bottle, 41=cup
COCO_CONF = 0.35

# TRT modunda hedef class (None = en yuksek skoru al)
TRT_TARGET_CLASS_ID = None
# trt_coco modunda: COCO_TARGET_CLASSES listesi kullanılır (yukarıda tanımlı).
# Yaw/Pitch sign (SiYi ↔ kamera konvansiyonu). Gimbal yanlış yöne dönerse -1 yap.
YAW_SIGN = 1
PITCH_SIGN = 1
# ────────────────────────────────────────────────────────────────

STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

PID_YAW_KP = 35.0
PID_YAW_KI = 1.0
PID_YAW_KD = 5.0
PID_PITCH_KP = 30.0
PID_PITCH_KI = 0.8
PID_PITCH_KD = 4.0

MAX_LOST_FRAMES = 30


class ThreadedCapture:
    """
    RTSP akışını arka planda sürekli okur, sadece en son frame'i saklar.
    Buffer birikimini engelleyerek gecikmeyi ortadan kaldırır.
    """
    def __init__(self, uri):
        self._cap = cv2.VideoCapture(uri)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._new_frame = False
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._new_frame = True

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            frame = self._frame.copy()
            self._new_frame = False
            return True, frame

    def isOpened(self):
        return self._cap.isOpened()

    def release(self):
        self._running = False
        self._cap.release()


class AutonomousTracker:
    def __init__(self):
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("=" * 60)
        print("  USV OTONOM HEDEF TAKIP SISTEMI")
        print("=" * 60)

        if DETECTOR_MODE == 'trt_coco':
            print(f"[INIT] YOLOv8 TRT COCO engine yukleniyor (su sisesi modu, ~30fps)...")
            from yolov8_trt import YOLOv8TRT
            self.detector = YOLOv8TRT(_coco_engine, COCO_CONF, IOU_THRESHOLD)
        elif DETECTOR_MODE == 'trt':
            print("[INIT] YOLOv8 TRT engine yukleniyor (duba modu)...")
            from yolov8_trt import YOLOv8TRT
            self.detector = YOLOv8TRT(ENGINE_PATH, CONF_THRESHOLD, IOU_THRESHOLD)
        else:
            print("[INIT] YOLOv8 COCO CPU modeli yukleniyor (yavas, TRT bekleniyor)...")
            from yolov8_coco import YOLOv8COCO
            self.detector = YOLOv8COCO(
                model_path=_coco_pt,
                conf_thresh=COCO_CONF,
                target_classes=COCO_TARGET_CLASSES,
            )

        print("[INIT] Gimbal baglantisi kuruluyor...")
        self.gimbal = SIYIGimbalController(GIMBAL_IP, GIMBAL_PORT)
        self.gimbal_connected = False
        fw = self.gimbal.get_firmware_version()
        if fw:
            print(f"[OK] Gimbal baglandi: {fw}")
            self.gimbal_connected = True
        else:
            print("[WARN] Gimbal yanit vermiyor! Sadece tespit modu.")

        self.pid_yaw = PIDController(kp=PID_YAW_KP, ki=PID_YAW_KI, kd=PID_YAW_KD)
        self.pid_pitch = PIDController(kp=PID_PITCH_KP, ki=PID_PITCH_KI, kd=PID_PITCH_KD)

        self.tracked_id = None
        self.lost_count = 0
        self.frame_count = 0
        self.fps_time = time.time()
        self.current_fps = 0.0
        self._last_cmd = (0, 0)

    def _signal_handler(self, signum, frame):
        print("\n[INFO] Kapatiliyor...")
        self.running = False

    def _draw_frame(self, frame, detections, target, error_x, error_y):
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx, cy = w // 2, h // 2

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            is_target = (det is target)
            color = (0, 0, 255) if is_target else (0, 255, 0)
            thick  = 3 if is_target else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
            label = f"{det['class_name']} {det['score']:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Merkez artisi
        cv2.line(vis, (cx - 35, cy), (cx + 35, cy), (0, 255, 255), 2)
        cv2.line(vis, (cx, cy - 35), (cx, cy + 35), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 6, (0, 255, 255), -1)

        # Hedefe cizgi
        if target:
            tx = int((target['bbox'][0] + target['bbox'][2]) / 2)
            ty = int((target['bbox'][1] + target['bbox'][3]) / 2)
            cv2.arrowedLine(vis, (cx, cy), (tx, ty), (0, 165, 255), 2, tipLength=0.15)

        # Sol ust: durum bilgisi
        def txt(img, s, y, color=(255, 255, 255)):
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4)
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        txt(vis, f"FPS: {self.current_fps:.1f}", 30)
        if self.gimbal_connected:
            txt(vis, f"Gimbal  Y:{self.gimbal.current_yaw:+.1f}  P:{self.gimbal.current_pitch:+.1f}", 62)
        if target:
            txt(vis, f"TAKIP: {target['class_name']}  %{target['score']*100:.0f}", 94, (0, 255, 0))
            txt(vis, f"Hata  X:{error_x:+.2f}  Y:{error_y:+.2f}", 124, (0, 220, 0))
        else:
            txt(vis, "ARAMA...", 94, (0, 165, 255))

        return vis

    def _open_rtsp(self):
        print(f"[INFO] RTSP baglantisi kuruluyor: {RTSP_URI}")
        cap = ThreadedCapture(RTSP_URI)
        if cap.isOpened():
            print("[OK] RTSP baglantisi basarili! (threaded, gecikme yok)")
            return cap
        print("[ERROR] Kamera acilamadi!")
        return None

    def _select_target(self, detections):
        if not detections:
            return None
        if DETECTOR_MODE == 'trt':
            allowed = {TRT_TARGET_CLASS_ID} if TRT_TARGET_CLASS_ID is not None else None
        elif DETECTOR_MODE == 'trt_coco':
            allowed = set(COCO_TARGET_CLASSES) if COCO_TARGET_CLASSES else None
        else:
            allowed = None  # coco detector zaten filtreliyor
        best, best_score = None, 0
        for det in detections:
            if allowed is not None and det['class_id'] not in allowed:
                continue
            if det['score'] > best_score:
                best_score = det['score']
                best = det
        return best

    def _track_target(self, detection):
        bbox = detection['bbox']
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        error_x = (cx - STREAM_WIDTH / 2.0) / (STREAM_WIDTH / 2.0)
        error_y = (cy - STREAM_HEIGHT / 2.0) / (STREAM_HEIGHT / 2.0)
        yaw_speed = self.pid_yaw.compute(error_x)
        pitch_speed = self.pid_pitch.compute(-error_y)
        y_cmd = int(yaw_speed) * YAW_SIGN
        p_cmd = int(pitch_speed) * PITCH_SIGN
        self._last_cmd = (y_cmd, p_cmd)
        try:
            self.gimbal.set_gimbal_rotation(y_cmd, p_cmd)
        except Exception:
            pass
        return error_x, error_y

    def _on_target_lost(self):
        self.tracked_id = None
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        if self.gimbal_connected:
            try:
                self.gimbal.set_gimbal_rotation(0, 0)
            except:
                pass

    def run(self):
        cap = self._open_rtsp()
        if cap is None:
            print("[ERROR] Kamera baglantisi kurulamadi! Cikiliyor.")
            return

        print("[INFO] Sistem baslatildi! Ctrl+C ile cikis.")

        if self.gimbal_connected:
            try:
                self.gimbal.center_gimbal()
                time.sleep(1)
                self.gimbal.start_attitude_polling(interval=0.1)
            except:
                print("[WARN] Gimbal merkeze alinamadi")

        no_frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                no_frame_count += 1
                if no_frame_count > 100:
                    print("[ERROR] Kamera sinyal vermiyor. Cikiliyor.")
                    break
                time.sleep(0.01)
                continue
            no_frame_count = 0

            detections = self.detector.detect(frame)
            target = self._select_target(detections)

            error_x, error_y = 0, 0
            if target:
                error_x, error_y = self._track_target(target)
                self.lost_count = 0
            else:
                self.lost_count += 1
                if self.lost_count > MAX_LOST_FRAMES:
                    self._on_target_lost()

            vis = self._draw_frame(frame, detections, target, error_x, error_y)
            cv2.imshow("USV Buoy Tracker", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False

            self.frame_count += 1
            elapsed = time.time() - self.fps_time
            if elapsed > 2.0:
                self.current_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_time = time.time()

                det_info = ""
                for d in detections:
                    det_info += f" {d['class_name']}({d['score']:.2f})"

                gimbal_info = ""
                if self.gimbal_connected:
                    gimbal_info = f" | Gimbal: Y={self.gimbal.current_yaw:.1f} P={self.gimbal.current_pitch:.1f}"

                target_info = ""
                if target:
                    ycmd, pcmd = self._last_cmd
                    target_info = f" | TARGET: {target['class_name']} err:({error_x:+.2f},{error_y:+.2f}) cmd:(Y={ycmd:+d},P={pcmd:+d})"
                else:
                    target_info = " | SEARCHING..."

                print(f"[INFO] FPS: {self.current_fps:.1f} | Det: {len(detections)}{det_info}{target_info}{gimbal_info}")

        print("[INFO] Temizlik yapiliyor...")
        cap.release()
        cv2.destroyAllWindows()
        if self.gimbal_connected:
            try:
                self.gimbal.stop()
            except:
                pass
        print("[INFO] Sistem kapatildi.")

if __name__ == "__main__":
    tracker = AutonomousTracker()
    tracker.run()
