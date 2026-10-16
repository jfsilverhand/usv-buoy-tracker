#!/bin/bash
# ============================================================
#  USV Buoy Tracker — x86 GPU'da DeepStream smoke test (Jetson gerekmez)
#
#  Amaç: ds_main.py pipeline'ını, DeepStream-Yolo parser derlemesini, ONNX → TRT
#  engine üretimini, nvinfer/nvtracker config'lerini gerçek bir DeepStream 7.1
#  kurulumunda doğrulamak. Jetson FPS'ini ÖLÇMEZ (farklı GPU), sadece "çalışıyor mu".
#
#  Nerede çalıştırılır: NVIDIA GPU'lu herhangi bir x86 Linux makinede,
#  nvcr.io/nvidia/deepstream:7.1-triton-multiarch container'ının İÇİNDE (nvcc/CUDA toolkit içerir).
#
#  vast.ai / RunPod ile (≈$0.2–0.5/saat):
#    1. Image: nvcr.io/nvidia/deepstream:7.1-triton-multiarch   (RTX 30xx/40xx yeter, disk ≥ 50 GB)
#    2. SSH ile gir, sonra:
#         git clone --recursive https://github.com/jfsilverhand/usv-buoy-tracker.git ~/usv-buoy-tracker
#         cd ~/usv-buoy-tracker && bash tools/x86_smoke_test.sh
#    3. Test videosu: videos/ gitignore'da. Kendi videonu scp ile ~/usv-buoy-tracker/videos/ altına at,
#       ya da VIDEO= ile başka bir dosya ver. Hiçbiri yoksa DeepStream'in örnek videosu kullanılır
#       (duba yok → tespit 0 ama pipeline çalışır).
#
#  Çıktı: eval/ds_smoke.mp4 (ilk 300 kare, OSD çizimli). Bunu indirip GIF'e çevirebilirsin.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
FRAMES="${FRAMES:-300}"
VIDEO="${VIDEO:-}"
DS_SAMPLE=/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4

step() { echo; echo "────────────────────────────────────────"; echo "  $*"; echo "────────────────────────────────────────"; }

step "[0/5] Ortam kontrolü"
[ -d /opt/nvidia/deepstream/deepstream ] || { echo "[ERROR] DeepStream yok. Bu script deepstream:7.1 container'ı içinde çalışır."; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || { echo "[ERROR] GPU görünmüyor"; exit 1; }
ARCH=$(uname -m); echo "arch: $ARCH"
if [ "$ARCH" != "x86_64" ]; then echo "[WARN] x86 değil ($ARCH). Jetson'daysan ./setup.sh kullan."; fi

step "[1/5] Python bağımlılıkları (gi, pyds, numpy, opencv)"
if ! python3 -c "import gi" 2>/dev/null; then
    apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        python3-gi python3-gi-cairo gir1.2-gst-rtsp-server-1.0 gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-ugly gstreamer1.0-libav python3-pip git build-essential wget >/dev/null
fi
if ! python3 -c "import pyds" 2>/dev/null; then
    PYV=$(python3 -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
    # pyds 1.1.11 = DeepStream 7.1; x86 wheel aynı release'te
    W=pyds-1.1.11-py3-none-linux_x86_64.whl
    wget -q "https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.11/$W" -O /tmp/$W
    pip3 install -q /tmp/$W && rm /tmp/$W
fi
pip3 install -q "numpy<2" opencv-python-headless onnx 2>/dev/null || true
python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; import pyds; print('gi + pyds OK')"

step "[2/5] DeepStream-Yolo parser (.so)"
if [ ! -f deepstream_yolo/nvdsinfer_custom_impl_Yolo/Makefile ]; then
    git submodule update --init --depth 1 deepstream_yolo 2>/dev/null || \
    git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo deepstream_yolo
fi
SO=deepstream_yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
if [ ! -f "$SO" ]; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
    CUDA_VER=${CUDA_VER:-$(ls -d /usr/local/cuda-* 2>/dev/null | head -1 | grep -oP '[0-9]+\.[0-9]+')}
    echo "CUDA_VER=$CUDA_VER"
    ( cd deepstream_yolo/nvdsinfer_custom_impl_Yolo && make -j"$(nproc)" CUDA_VER="$CUDA_VER" )
fi
ls -la "$SO"

step "[3/5] Model kontrolü"
[ -f models/best_ds.onnx ] || { echo "[ERROR] models/best_ds.onnx yok → tools/export_for_deepstream.sh models/best.pt"; exit 1; }
python3 - <<'EOF'
import onnx; m = onnx.load("models/best_ds.onnx")
outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
print("outputs:", outs)
assert outs[0][1][-1] == 6, "best_ds.onnx DeepStream-Yolo formatında değil (son boyut 6 olmalı)"
EOF
rm -f model_b1_gpu0_fp16.engine   # eski yöntemin artığı
[ "${REBUILD:-0}" = "1" ] && rm -f models/best_ds*.engine && echo "engine silindi, yeniden derlenecek"

step "[4/5] Test videosu"
if [ -z "$VIDEO" ]; then
    VIDEO=$(ls videos/*.mp4 2>/dev/null | head -1 || true)
fi
if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
    echo "[WARN] videos/ altında mp4 yok, DeepStream örnek videosu kullanılıyor (duba yok, tespit 0 normal)."
    VIDEO=$DS_SAMPLE
fi
echo "video: $VIDEO"

step "[5/5] ds_main.py — $FRAMES kare, headless, mp4'e yaz (ilk çalıştırmada TRT engine ~3-5 dk)"
mkdir -p eval
set +e
python3 src/ds_main.py --src "$VIDEO" --sink file --out eval/ds_smoke.mp4 --max-frames "$FRAMES" --no-gimbal 2>&1 | tee eval/ds_smoke.log
RC=${PIPESTATUS[0]}
set -e

echo
if [ "$RC" -eq 0 ] && [ -s eval/ds_smoke.mp4 ]; then
    echo "✅ SMOKE TEST GEÇTİ  → eval/ds_smoke.mp4 ($(du -h eval/ds_smoke.mp4 | cut -f1)), log: eval/ds_smoke.log"
    echo "   Engine: $(ls -la models/best_ds.onnx_b1_gpu0_fp16.engine 2>/dev/null | awk '{print $5" bytes"}')"
    grep "\[SUMMARY\]" eval/ds_smoke.log | sed "s/^/   /"
else
    echo "❌ SMOKE TEST BAŞARISIZ (rc=$RC). eval/ds_smoke.log'un son 40 satırı:"
    tail -40 eval/ds_smoke.log
    exit 1
fi
