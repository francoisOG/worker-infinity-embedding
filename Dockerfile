FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV HF_HOME=/runpod-volume

# install python and other packages
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# install uv
RUN pip install uv

# install python dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt --system

# install torch
RUN pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

# --- OPTIONAL: pre-bake the embedding model into the image to remove the
# Hugging Face download from cold start.
# NOTE: HF_HOME above is /runpod-volume, a RUNTIME network volume that SHADOWS
# anything baked at that path during build. To truly bake the model into the
# image, download it to an image-local path and point HF_HOME there instead —
# uncomment the two lines below AND comment out the `ENV HF_HOME=/runpod-volume`
# line at the top. (Keep the volume cache instead if you prefer download-once-
# to-volume rather than a larger image.)
# ENV HF_HOME=/models
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B')"

# Add src files
ADD src .

# Add test input
COPY test_input.json /test_input.json

# start the handler
CMD python -u /handler.py
