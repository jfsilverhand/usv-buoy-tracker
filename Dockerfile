FROM nvcr.io/nvidia/deepstream-l4t:7.1-samples-multiarch

ENV DEBIAN_FRONTEND=noninteractive

# GStreamer Python bindings + build araçları (DeepStream-Yolo parser derlemek için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gst-rtsp-server-1.0 \
    wget \
    git \
    build-essential \
    libpython3.10 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-x \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# DeepStream 7.1 Python bindings (pyds)
RUN wget -q https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.11/pyds-1.1.11-py3-none-linux_aarch64.whl \
    && pip3 install pyds-1.1.11-py3-none-linux_aarch64.whl \
    && rm pyds-1.1.11-py3-none-linux_aarch64.whl

# NumPy (pyds/TensorRT ile uyumlu 1.x) + OpenCV (aarch64 manylinux wheel, ffmpeg dahili) + pyserial
RUN pip3 install --no-cache-dir \
    "numpy==1.26.4" \
    "opencv-python==4.10.0.84" \
    pyserial

# PyCUDA — wheels/ içinde önceden derlenmiş aarch64 wheel varsa onu kur,
# yoksa pip'ten derle (container'da nvcc var; ~10 dk sürer)
COPY wheels/ /tmp/wheels/
RUN if ls /tmp/wheels/pycuda*.whl >/dev/null 2>&1; then \
        pip3 install --no-cache-dir /tmp/wheels/pycuda*.whl; \
    else \
        pip3 install --no-cache-dir pycuda; \
    fi && rm -rf /tmp/wheels

# Kaynak kod + configler (prod için; dev'de ./:/workspace mount override eder)
COPY src/ /workspace/src/
COPY configs/ /workspace/configs/

# COCO demo ONNX — proje klasörü mount edilmese bile image içinde sabit yerde
COPY models/yolov8n_coco.onnx /opt/ida/yolov8n_coco.onnx

# Entrypoint: ilk çalıştırmada vision_test için TRT engine build eder
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "src/vision_test.py"]
