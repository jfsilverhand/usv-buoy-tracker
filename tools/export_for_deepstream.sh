#!/bin/bash
# ============================================================
#  YOLOv8 .pt → DeepStream-Yolo uyumlu ONNX
#
#  NvDsInferParseYolo, Ultralytics'in standart export'unu ([1, 4+nc, 8400])
#  okuyamaz; boxes/scores/classes çıkışlı özel export ister. Bu script
#  deepstream_yolo/utils/export_yoloV8.py'yi çalıştırıp çıktıyı models/ altına
#  <isim>_ds.onnx olarak koyar. PC'de (x86) çalıştırılabilir, Jetson gerekmez.
#
#  Kullanım:
#    tools/export_for_deepstream.sh models/best.pt          → models/best_ds.onnx
#    tools/export_for_deepstream.sh yolov8n.pt              → models/yolov8n_coco_ds.onnx
#  Gereksinim: pip install ultralytics onnx onnxslim onnxruntime
# ============================================================
set -e

WEIGHTS="${1:?Kullanım: $0 <model.pt>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_PY="$ROOT/deepstream_yolo/utils/export_yoloV8.py"

[ -f "$WEIGHTS" ]   || { echo "[ERROR] $WEIGHTS yok"; exit 1; }
[ -f "$EXPORT_PY" ] || { echo "[ERROR] $EXPORT_PY yok — önce ./setup.sh (adım 1) veya git submodule update --init"; exit 1; }
python3 -c "import ultralytics, onnx" 2>/dev/null || { echo "[ERROR] pip install ultralytics onnx onnxslim onnxruntime"; exit 1; }

BASENAME="$(basename "$WEIGHTS" .pt)"
case "$BASENAME" in
    yolov8n|yolov8s|yolov8m) OUT="$ROOT/models/${BASENAME}_coco_ds.onnx" ;;
    *)                       OUT="$ROOT/models/${BASENAME}_ds.onnx" ;;
esac

TMP="$(mktemp -d)"
cp "$WEIGHTS" "$TMP/"
cp "$EXPORT_PY" "$TMP/"
sed -i "s/opset_version=/dynamo=False, opset_version=/" "$TMP/export_yoloV8.py"
cd "$TMP"
echo "[EXPORT] $WEIGHTS → DeepStream-Yolo ONNX (opset 12, static batch 1, 640x640)"
python3 export_yoloV8.py -w "$(basename "$WEIGHTS")" --simplify --opset 12 --batch 1

ONNX_OUT="$(ls *.onnx | head -1)"
mkdir -p "$ROOT/models"
mv "$ONNX_OUT" "$OUT"
[ -f labels.txt ] && echo "[INFO] Export labels: $(tr '\n' ' ' < labels.txt)"
cd / && rm -rf "$TMP"

echo "[OK] $OUT"
echo "     nvinfer config'de onnx-file=$OUT olmalı; TRT engine Jetson'da ilk çalıştırmada üretilir."
