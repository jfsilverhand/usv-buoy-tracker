#!/bin/bash
# ============================================================
#  USV Buoy Tracker — Bir Defalık Kurulum (Jetson)
#  Yeni bir makinada: git clone --recursive <repo> && ./setup.sh
#  Image'ı yerelde build etmek için: BUILD=1 ./setup.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "  USV Buoy Tracker Kurulumu"
echo "=================================================="

# ── 1. DeepStream-Yolo parser kaynak kodu ─────────────────────────────────────
echo ""
echo "[1/5] DeepStream-Yolo kaynak kodu kontrol ediliyor..."
if [ -f deepstream_yolo/nvdsinfer_custom_impl_Yolo/Makefile ]; then
    echo "[OK] deepstream_yolo/ mevcut."
elif [ -f .gitmodules ] && grep -q deepstream_yolo .gitmodules; then
    echo "[GET] git submodule update --init"
    git submodule update --init --depth 1 deepstream_yolo
else
    echo "[GET] marcoslucianops/DeepStream-Yolo klonlanıyor..."
    rm -rf deepstream_yolo
    git clone --depth 1 https://github.com/marcoslucianops/DeepStream-Yolo deepstream_yolo
fi

# ── 2. Docker image ──────────────────────────────────────────────────────────
echo ""
echo "[2/5] Docker image hazırlanıyor..."
if [ "${BUILD:-0}" = "1" ]; then
    docker compose build
else
    docker compose pull || { echo "[WARN] pull başarısız, yerelde build ediliyor..."; docker compose build; }
fi
echo "[OK] İmaj hazır."

# ── 3. Parser .so (container içinde derlenir, bind mount sayesinde host'ta kalır) ──
echo ""
echo "[3/5] Parser kütüphanesi kontrol ediliyor..."
SO_PATH="deepstream_yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
if [ -f "$SO_PATH" ]; then
    echo "[OK] libnvdsinfer_custom_impl_Yolo.so mevcut, build atlanıyor."
else
    echo "[BUILD] Parser .so container içinde derleniyor..."
    docker compose run --rm --no-deps tracker bash -c '
        CUDA_VER=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
        CUDA_VER=${CUDA_VER:-12.6}
        echo "[INFO] CUDA version: $CUDA_VER"
        cd /workspace/deepstream_yolo/nvdsinfer_custom_impl_Yolo
        make clean >/dev/null 2>&1 || true
        make -j$(nproc) CUDA_VER=$CUDA_VER
        echo "[OK] Parser derlendi."
    '
fi

# ── 4. Tracker config (DS 5.x formatıysa DS 7.1 örneğiyle değiştir) ───────────
echo ""
echo "[4/5] Tracker config kontrol ediliyor..."
if grep -q "^NvDCF:" configs/tracker_config.yml 2>/dev/null; then
    echo "[FIX] Eski NvDCF formatı bulundu, DeepStream 7.1 örneği kopyalanıyor..."
    docker compose run --rm --no-deps tracker bash -c '
        cp /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml \
           /workspace/configs/tracker_config.yml'
fi
echo "[OK] Tracker config hazır."

# ── 5. Modeller ───────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Modeller kontrol ediliyor..."
if [ ! -f models/best_ds.onnx ]; then
    echo "[WARN] models/best_ds.onnx yok."
    echo "       DeepStream parser'ı, Ultralytics'in standart export'unu ([1,9,8400]) okuyamaz."
    echo "       Duba modelini DeepStream-Yolo export'uyla yeniden üret:"
    echo "         tools/export_for_deepstream.sh models/best.pt"
    echo "       (best.pt yoksa modeli eğiten kişiden iste.) ds_main.py bu dosya olmadan başlamaz."
else
    echo "[OK] models/best_ds.onnx mevcut. TRT engine'i nvinfer ilk çalıştırmada üretir (~5 dk)."
fi

# vision_test.py için COCO engine (entrypoint derler, ./models'a yazılır)
echo "[BUILD] COCO TRT engine (vision_test) kontrol ediliyor..."
docker compose run --rm --no-deps tracker true 2>&1 | tail -5

echo ""
echo "=================================================="
echo "  Kurulum tamamlandı!"
echo "  DeepStream takip : ./run.sh            (src/ds_main.py)"
echo "  Vision test      : ./run.sh test       (src/vision_test.py)"
echo "=================================================="
