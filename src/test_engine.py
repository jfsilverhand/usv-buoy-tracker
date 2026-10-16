#!/usr/bin/env python3
"""
Kamera olmadan YOLOv8 TensorRT engine testi (PyCUDA yolu, DeepStream değil).

  python3 src/test_engine.py                                  # models/yolov8n_coco_docker_fp16.engine
  python3 src/test_engine.py models/best_fp16_host.engine     # başka engine
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolov8_trt import YOLOv8TRT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_in_docker = os.path.exists("/.dockerenv")
DEFAULT_ENGINE = os.path.join(BASE_DIR, "models",
                              "yolov8n_coco_docker_fp16.engine" if _in_docker else "yolov8n_coco_fp16.engine")

engine_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ENGINE
if not os.path.exists(engine_path):
    sys.exit(f"[ERROR] Engine yok: {engine_path}\n"
             f"        Docker'da entrypoint.sh üretir; host'ta: trtexec --onnx=... --saveEngine={engine_path} --fp16")

detector = YOLOv8TRT(engine_path)
fake_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

print("\n[TEST] 100 frame inference testi basliyor...")
times = []
for i in range(100):
    t1 = time.time()
    results = detector.detect(fake_frame)
    t2 = time.time()
    times.append(t2 - t1)
    if i == 0:
        print(f"  Ilk frame: {len(results)} tespit, {(t2-t1)*1000:.1f}ms")

times = times[10:]
avg = np.mean(times) * 1000
fps = 1000 / avg
print(f"\n[SONUC] Ortalama inference (pre+infer+post): {avg:.1f}ms ({fps:.1f} FPS)")
print("[OK] Engine testi basarili!")
