"""
Ultralytics YOLOv8 ile COCO nesne tespiti — su sisesi ve diger genel nesneler
"""
from ultralytics import YOLO

# COCO'da su sisesi class 39
BOTTLE_CLASS_ID = 39

class YOLOv8COCO:
    def __init__(self, model_path='yolov8n.pt', conf_thresh=0.4, target_classes=None):
        """
        target_classes: None = hepsini tespit et, [39] = sadece bottle, vb.
        """
        self.conf_thresh = conf_thresh
        self.target_classes = target_classes
        print(f"[YOLOv8-COCO] Model yukleniyor: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        # Warmup
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, conf=self.conf_thresh, verbose=False, device='cpu')
        targets = target_classes if target_classes else 'tümü'
        print(f"[YOLOv8-COCO] Hazir | hedef class: {targets}")

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf_thresh,
            verbose=False,
            device='cpu',
            imgsz=320,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                if self.target_classes is not None and class_id not in self.target_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': float(box.conf[0]),
                    'class_id': class_id,
                    'class_name': self.class_names[class_id],
                })
        return detections
