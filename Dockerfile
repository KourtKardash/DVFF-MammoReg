# Registration image: PyTorch 2.4.0, CUDA 12.4.
# Build from the repository root:
#   docker build -t dvff-registration:cu124 .

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    WANDB_MODE=disabled

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace
