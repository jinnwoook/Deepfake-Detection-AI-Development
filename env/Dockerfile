FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git wget \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN pip install --upgrade pip

WORKDIR /app
COPY env/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Pre-download models for offline inference
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('OwensLab/commfor-model-384')"
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('arnabdhar/YOLOv8-Face-Detection', 'model.pt')"

ENTRYPOINT ["python", "inference.py"]
