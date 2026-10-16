"""
YOLOv8 TensorRT Inference Engine
"""
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush'
]

DUBA_NAMES = ['kirmiziduba', 'sariduba', 'siyahduba', 'turuncuduba', 'yesilduba']

class YOLOv8TRT:
    def __init__(self, engine_path, conf_thresh=0.25, iou_thresh=0.45, class_names=None):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # class_names None ise output shape'e göre otomatik seçilir
        self._custom_class_names = class_names
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape  = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape) * 4))
        self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape) * 4))
        # pinned memory: H→D transfer DMA ile direk, daha hızlı
        self.h_input  = cuda.pagelocked_empty(self.input_shape, dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(self.output_shape, dtype=np.float32)
        self.stream = cuda.Stream()
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        # TRT warmup — h_input zaten sıfır (pagelocked_empty garanti vermez, explicit sıfırla)
        self.h_input[:] = 0.0
        for _ in range(3):
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async_v3(self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
        # class_names: custom > COCO (84 çıktı) > duba (9 çıktı)
        num_classes = self.output_shape[1] - 4 if len(self.output_shape) > 1 else 5
        if self._custom_class_names:
            self.class_names = self._custom_class_names
        elif num_classes == 80:
            self.class_names = COCO_NAMES
        else:
            self.class_names = DUBA_NAMES
        # Preprocessing için sabit boyutlu buffer'lar (her frame'de alloc yok)
        self._pad_buf = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        self._prep_cache = {}  # giriş boyutuna göre scale/pad cache
        print(f"[YOLOv8-TRT] Engine loaded: {engine_path}")
        print(f"[YOLOv8-TRT] Input: {self.input_shape}, Output: {self.output_shape}, Classes: {len(self.class_names)}")

    def preprocess(self, frame):
        img_h, img_w = frame.shape[:2]

        # Aynı giriş boyutu için scale/pad değerlerini cache'le (tekrar hesaplama yok)
        key = (img_w, img_h)
        if key not in self._prep_cache:
            scale = min(self.input_w / img_w, self.input_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            pad_w = (self.input_w - new_w) // 2
            pad_h = (self.input_h - new_h) // 2
            self._prep_cache[key] = (scale, new_w, new_h, pad_w, pad_h)
        scale, new_w, new_h, pad_w, pad_h = self._prep_cache[key]

        # resize → pad_buf'a yaz (alloc yok), padding bölgeleri zaten 114 ile dolu
        dst = self._pad_buf[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
        cv2.resize(frame, (new_w, new_h), dst=dst)

        # uint8 HWC → float32 CHW, direkt pinned h_input'a yaz (intermediate array yok)
        for c in range(3):
            np.multiply(self._pad_buf[:, :, c], 1.0 / 255.0,
                        out=self.h_input[0, c])

        return scale, pad_w, pad_h

    def postprocess(self, output, scale, pad_w, pad_h, img_w, img_h):
        output = output[0]
        predictions = output.T
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        mask = max_scores > self.conf_thresh
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        if len(boxes) == 0:
            return []
        cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1 = (cx - w/2 - pad_w) / scale
        y1 = (cy - h/2 - pad_h) / scale
        x2 = (cx + w/2 - pad_w) / scale
        y2 = (cy + h/2 - pad_h) / scale
        x1 = np.clip(x1, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)
        bxs = np.stack([x1, y1, x2, y2], axis=1)
        indices = self.nms(bxs, max_scores, self.iou_thresh)
        results = []
        for i in indices:
            results.append({
                'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                'score': float(max_scores[i]),
                'class_id': int(class_ids[i]),
                'class_name': self.class_names[int(class_ids[i])]
            })
        return results

    def nms(self, boxes, scores, iou_thresh):
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    def detect(self, frame):
        img_h, img_w = frame.shape[:2]
        scale, pad_w, pad_h = self.preprocess(frame)  # h_input'a direkt yazar
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.postprocess(self.h_output, scale, pad_w, pad_h, img_w, img_h)
