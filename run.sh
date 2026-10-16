#!/bin/bash
# ============================================================
#  USV Buoy Tracker — Çalıştır
#    ./run.sh          → DeepStream takip (src/ds_main.py)
#    ./run.sh test     → vision_test.py (COCO benchmark, RTSP)
#    ./run.sh test --src videos/x.mp4   → vision_test video dosyasıyla
#  Ortam değişkenleri: RTSP_URI, GIMBAL_IP, DISPLAY
# ============================================================
set -e

IMAGE="erengwen/usv-buoy-tracker:latest"
RTSP="${RTSP_URI:-rtsp://192.168.2.25:8554/main.264}"
GIMBAL="${GIMBAL_IP:-192.168.2.25}"
DISPLAY_VAL="${DISPLAY:-:0}"

MODE="${1:-track}"
[ $# -gt 0 ] && shift

# X11 erişimi
xhost +local:docker 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/src" ]; then
    # Proje klasöründeyiz: her şey (src, configs, models, parser .so) bind mount
    MOUNT="-v ${SCRIPT_DIR}:/workspace"
else
    # Sadece run.sh indirilmiş: engine'ler named volume'da kalsın
    MOUNT="-v ida_models:/workspace/models"
fi

case "$MODE" in
    track) CMD="python3 src/ds_main.py" ;;
    test)  CMD="python3 src/vision_test.py --src $RTSP" ;;
    *)     echo "Kullanım: ./run.sh [track|test] [ek argümanlar]"; exit 1 ;;
esac

echo "[INFO] USV Buoy Tracker başlatılıyor ($MODE)"
echo "[INFO] RTSP: $RTSP | Gimbal: $GIMBAL | DISPLAY: $DISPLAY_VAL"

docker run --rm -it \
    --runtime=nvidia \
    --privileged \
    --network=host \
    --ipc=host \
    -e DISPLAY="$DISPLAY_VAL" \
    -e XAUTHORITY=/root/.Xauthority \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e EGL_PLATFORM=surfaceless \
    -e RTSP_URI="$RTSP" \
    -e GIMBAL_IP="$GIMBAL" \
    --device /dev/dri:/dev/dri \
    -v "${HOME}/.Xauthority:/root/.Xauthority:ro" \
    $MOUNT \
    "$IMAGE" \
    $CMD "$@"
