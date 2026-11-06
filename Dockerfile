FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EASYOCR_MODEL_DIR=/models \
    DATA_DIR=/data

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# CUDA 12.8 PyTorch wheels support NVIDIA Blackwell (RTX 50-series sm_120) as well as older GPUs.
# Override with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu for CPU-only hosts.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install --index-url ${TORCH_INDEX} torch torchvision

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && pip uninstall -y opencv-python || true \
 && pip install --force-reinstall --no-deps opencv-python-headless

# Bake the English OCR models into the image so runs don't re-download them
RUN mkdir -p /models /data \
 && python -c "import easyocr; easyocr.Reader(['en'], gpu=False, model_storage_directory='/models')"

WORKDIR /app
COPY video_cleanse.py /app/video_cleanse.py

VOLUME ["/data"]
ENTRYPOINT ["python", "/app/video_cleanse.py"]
