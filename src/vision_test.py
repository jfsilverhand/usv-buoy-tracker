#!/usr/bin/env python3
"""
COCO modeli ile gerçek zamanlı görüntü işleme performans testi.
GStreamer nvv4l2decoder (GPU H.264 decode) kullanır → düşük gecikme, yüksek FPS.
Host ve Docker'da çalışır.

Kullanım:
  python3 vision_test.py                        # RTSP (gimbal kamera)
  python3 vision_test.py --src /path/video.mp4  # video dosyası
  python3 vision_test.py --src 0                # USB webcam

Tuşlar:  q=çıkış  +/-=confidence  s=istatistik  p=duraklat
"""

import cv2
import numpy as np
import time
import sys
import os
import argparse
import collections
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RTSP_URI    = os.environ.get("RTSP_URI", "rtsp://192.168.2.25:8554/main.264")
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# TRT engine cihaza + TensorRT sürümüne özeldir: Docker içinde entrypoint.sh'ın ürettiği,
# host'ta elle üretilen engine kullanılır (/.dockerenv varsa Docker içindeyiz).
_IN_DOCKER = os.path.exists("/.dockerenv")
_ENGINE_NAME = "yolov8n_coco_docker_fp16.engine" if _IN_DOCKER else "yolov8n_coco_fp16.engine"
COCO_ENGINE = os.environ.get("COCO_ENGINE", os.path.join(BASE_DIR, "models", _ENGINE_NAME))
COCO_PT     = os.path.join(BASE_DIR, "src", "yolov8n.pt")   # sadece CPU fallback (ultralytics gerekir, image'da yok)

CONF_THRESH = 0.35
IOU_THRESH  = 0.45
WINDOW_SIZE = 60


# ─── GStreamer tabanlı capture (GPU H.264 decode) ───────────────────────────

class GstCapture:
    """
    nvv4l2decoder ile GPU H.264 decode → CPU H.264 yerine ~15ms kazanım.
    appsink max-buffers=1 + drop=true: eski frame'ler atılır, gecikme sıfır.
    Hem host hem Docker'da çalışır (python3-gi gerekli).
    """
    def __init__(self, rtsp_uri, width=1280, height=720):
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
        self._Gst = Gst

        pipe_str = (
            f'rtspsrc location={rtsp_uri} latency=100 protocols=tcp ! '
            'rtph264depay ! h264parse ! '
            'nvv4l2decoder ! nvvidconv ! '
            f'video/x-raw,format=BGRx,width={width},height={height} ! '
            'videoconvert ! '
            'video/x-raw,format=BGR ! '
            'appsink name=sink drop=true max-buffers=1 sync=false emit-signals=false'
        )
        self._pipeline = Gst.parse_launch(pipe_str)
        self._sink     = self._pipeline.get_by_name('sink')
        self._pipeline.set_state(Gst.State.PLAYING)

        self.width  = width
        self.height = height
        self._frame = None
        self._lock  = threading.Lock()
        self._running = True
        self._seq   = 0  # her yeni frame'de artar

        threading.Thread(target=self._reader, daemon=True).start()
        time.sleep(1.5)  # pipeline warmup

    def _reader(self):
        Gst = self._Gst
        while self._running:
            sample = self._sink.emit('pull-sample')
            if sample is None:
                time.sleep(0.002)
                continue
            buf  = sample.get_buffer()
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if ok:
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(
                    self.height, self.width, 3).copy()
                buf.unmap(mapinfo)
                with self._lock:
                    self._frame = arr
                    self._seq  += 1

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None, -1
            return True, self._frame, self._seq

    def isOpened(self):
        return self._pipeline is not None

    def release(self):
        self._running = False
        self._pipeline.set_state(self._Gst.State.NULL)


# ─── Fallback: cv2.VideoCapture tabanlı threaded capture ────────────────────

class Cv2Capture:
    """cv2.VideoCapture fallback — GStreamer yoksa kullanılır."""
    def __init__(self, src):
        try:
            src = int(src)
        except (ValueError, TypeError):
            pass
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap   = cap
        self._frame = None
        self._seq   = 0
        self._lock  = threading.Lock()
        self._running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._seq  += 1

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None, -1
            return True, self._frame.copy(), self._seq

    def isOpened(self):
        return self._cap.isOpened()

    def release(self):
        self._running = False
        self._cap.release()


def open_capture(src):
    """GstCapture dene, başarısız olursa Cv2Capture kullan."""
    if isinstance(src, str) and src.startswith('rtsp'):
        try:
            cap = GstCapture(src)
            print("[CAPTURE] GStreamer nvv4l2decoder (GPU decode) aktif")
            return cap, 'GStreamer'
        except Exception as e:
            print(f"[CAPTURE] GStreamer başlatılamadı ({e}), cv2 fallback...")

    cap = Cv2Capture(src)
    if not cap.isOpened():
        return None, ''
    print("[CAPTURE] cv2.VideoCapture (CPU decode) aktif")
    return cap, 'cv2'


# ─── Rolling istatistik ──────────────────────────────────────────────────────

class RollingStats:
    def __init__(self, n=WINDOW_SIZE):
        self.buf = collections.deque(maxlen=n)

    def add(self, v):   self.buf.append(v)
    def avg(self):      return sum(self.buf) / len(self.buf) if self.buf else 0
    def minimum(self):  return min(self.buf) if self.buf else 0
    def maximum(self):  return max(self.buf) if self.buf else 0
    def p95(self):
        if not self.buf: return 0
        return sorted(self.buf)[int(len(self.buf) * 0.95)]


# ─── Detector yükle ─────────────────────────────────────────────────────────

def load_detector(engine_path, conf, iou):
    if os.path.exists(engine_path):
        print(f"[DETECTOR] TRT engine: {engine_path}")
        from yolov8_trt import YOLOv8TRT
        det = YOLOv8TRT(engine_path, conf, iou)
        return det, 'TRT-FP16'
    print("[DETECTOR] TRT engine yok, Ultralytics CPU fallback...")
    from yolov8_coco import YOLOv8COCO
    return YOLOv8COCO(model_path=COCO_PT, conf_thresh=conf), 'CPU'


# ─── Görsel overlay ─────────────────────────────────────────────────────────

def draw_overlay(frame, detections, metrics, conf, target_classes, paused, cap_mode):
    vis = frame.copy()
    h, w = vis.shape[:2]

    for det in detections:
        if target_classes and det['class_id'] not in target_classes:
            continue
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det['class_name']} {det['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, max(y1-th-6, 0)), (x1+tw+4, y1), (0, 200, 0), -1)
        cv2.putText(vis, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Merkez artı işareti
    cx, cy = w // 2, h // 2
    cv2.line(vis, (cx-30, cy), (cx+30, cy), (0, 255, 255), 1)
    cv2.line(vis, (cx, cy-30), (cx, cy+30), (0, 255, 255), 1)

    def txt(img, s, y, col=(220, 220, 220)):
        cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3)
        cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1)

    fps_col = (0, 255, 100) if metrics['fps'] > 25 else (0, 165, 255)
    txt(vis, f"FPS : {metrics['fps']:.1f}  (min {metrics['fps_min']:.0f} / p95 {metrics['fps_p95']:.0f})", 28, fps_col)
    txt(vis, f"TRT : {metrics['infer_ms']:.1f} ms  (avg {metrics['infer_avg']:.1f} ms)", 54)
    txt(vis, f"DEC : {cap_mode}  |  {w}x{h}", 80)
    txt(vis, f"Det : {metrics['det_count']}  Conf: {conf:.2f}  (+/- degistir)", 106)
    if paused:
        txt(vis, "[ DURAKLADI ]", 132, (0, 165, 255))

    # Sağ üst: class sayacı
    items = sorted(metrics['class_counts'].items(), key=lambda x: -x[1])[:8]
    if items:
        bx = w - 210
        overlay = vis.copy()
        cv2.rectangle(overlay, (bx-4, 4), (w-4, len(items)*22+14), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
        for i, (name, cnt) in enumerate(items):
            cv2.putText(vis, f"{name}: {cnt}", (bx, 22+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 210, 80), 1)

    return vis


# ─── Ana döngü ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=RTSP_URI)
    p.add_argument("--conf", type=float, default=CONF_THRESH)
    p.add_argument("--classes", nargs="*", type=int, default=None,
                   help="COCO class ID'leri (orn: 0 39 41). None=hepsi")
    return p.parse_args()


def main():
    args   = parse_args()
    conf   = args.conf
    target = args.classes

    detector, det_backend = load_detector(COCO_ENGINE, conf, IOU_THRESH)
    cap, cap_mode = open_capture(args.src)
    if cap is None:
        print(f"[ERROR] Kaynak açılamadı: {args.src}")
        return

    print(f"[INFO] {cap.width}x{cap.height} | Detector: {det_backend} | Capture: {cap_mode}")
    print("[INFO] Tuşlar: q=çıkış  +/-=conf  s=istatistik  p=duraklat")

    fps_st   = RollingStats()
    inf_st   = RollingStats()
    class_ct = collections.Counter()

    frame_count = 0
    paused      = False
    last_frame  = None
    last_dets   = []
    last_seq    = -1

    metrics = dict(fps=0, fps_min=0, fps_p95=0,
                   infer_ms=0, infer_avg=0, det_count=0, class_counts={})
    prev_t = time.perf_counter()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            conf = min(conf + 0.05, 0.95)
            print(f"[CONF] {conf:.2f}")
        elif key == ord('-'):
            conf = max(conf - 0.05, 0.05)
            print(f"[CONF] {conf:.2f}")
        elif key == ord('s'):
            print(f"\n=== ISTATISTIK (frame {frame_count}) ===")
            print(f"  FPS  avg={fps_st.avg():.1f}  min={fps_st.minimum():.1f}  p95={fps_st.p95():.1f}")
            print(f"  TRT  avg={inf_st.avg():.1f}ms  p95={inf_st.p95():.1f}ms")
            print(f"  Sinif: {dict(class_ct.most_common(10))}")
            print()
        elif key == ord('p'):
            paused = not paused

        if paused:
            if last_frame is not None:
                cv2.imshow("USV Vision Test", draw_overlay(
                    last_frame, last_dets, metrics, conf, target, True, cap_mode))
            continue

        ret, frame, seq = cap.read()
        if not ret or frame is None:
            time.sleep(0.005)
            continue

        # Kamera FPS'i < inference FPS ise aynı frame'i tekrar işleme
        if seq == last_seq:
            time.sleep(0.001)
            if last_frame is not None:
                cv2.imshow("USV Vision Test", draw_overlay(
                    last_frame, last_dets, metrics, conf, target, False, cap_mode))
            continue
        last_seq   = seq
        last_frame = frame

        detector.conf_thresh = conf
        t0    = time.perf_counter()
        dets  = detector.detect(frame)
        t_inf = (time.perf_counter() - t0) * 1000
        last_dets = dets

        now  = time.perf_counter()
        fps  = 1.0 / (now - prev_t) if (now - prev_t) > 0 else 0
        prev_t = now

        fps_st.add(fps)
        inf_st.add(t_inf)
        for d in dets:
            if target is None or d['class_id'] in target:
                class_ct[d['class_name']] += 1
        frame_count += 1

        metrics.update(
            fps=fps_st.avg(), fps_min=fps_st.minimum(), fps_p95=fps_st.p95(),
            infer_ms=t_inf, infer_avg=inf_st.avg(),
            det_count=sum(1 for d in dets if target is None or d['class_id'] in target),
            class_counts=dict(class_ct.most_common(10)),
        )

        cv2.imshow("USV Vision Test", draw_overlay(
            frame, dets, metrics, conf, target, False, cap_mode))

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n=== OZET ({frame_count} frame) ===")
    print(f"FPS   avg={fps_st.avg():.1f}  min={fps_st.minimum():.1f}  p95={fps_st.p95():.1f}")
    print(f"TRT   avg={inf_st.avg():.1f}ms  p95={inf_st.p95():.1f}ms")
    print(f"Sinif: {dict(class_ct.most_common())}")


if __name__ == "__main__":
    main()
