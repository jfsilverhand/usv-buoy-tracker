#!/bin/bash
# İlk çalıştırmada vision_test.py / main.py için COCO TRT engine'ini derler.
# DeepStream (ds_main.py) engine'lerini nvinfer kendisi onnx-file'dan üretir.
set -e

ENGINE=/workspace/models/yolov8n_coco_docker_fp16.engine
ONNX=/workspace/models/yolov8n_coco.onnx
[ -f "$ONNX" ] || ONNX=/opt/ida/yolov8n_coco.onnx   # proje mount edilmediyse image içindeki kopya

if [ ! -f "$ENGINE" ]; then
    echo "=================================================="
    echo "  IDA Tracker — İlk Çalıştırma"
    echo "  TRT engine derleniyor (~5 dakika, bir kez yapılır)"
    echo "  $ONNX -> $ENGINE"
    echo "=================================================="
    mkdir -p /workspace/models
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$ONNX" \
        --saveEngine="$ENGINE" \
        --fp16 2>&1 | grep -E "PASS|FAIL|Timing|Mean|error|Error" | tail -8
    echo "[OK] Engine hazır: $ENGINE"
    echo "=================================================="
fi

exec "$@"
