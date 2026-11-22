FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/tracking \
    UV_NO_CACHE=1 \
    SAM2_BUILD_CUDA=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.11

WORKDIR /app
COPY pyproject.toml uv.lock ./
ENV UV_NO_CACHE=0 UV_LINK_MODE=copy

# Fetch only the SAM 2 package and its setup files from the locked revision.
RUN <<'SH'
set -eu
python="$(uv python find 3.11)"
sam2_source="$("$python" - <<'PY'
import tomllib

with open("uv.lock", "rb") as lock_file:
    packages = tomllib.load(lock_file)["package"]
sam2 = next(package for package in packages if package["name"] == "sam-2")
print(sam2["source"]["git"])
PY
)"

git init --quiet /opt/sam2
git -C /opt/sam2 remote add origin "${sam2_source%#*}"
git -C /opt/sam2 config remote.origin.promisor true
git -C /opt/sam2 config remote.origin.partialclonefilter blob:none
git -C /opt/sam2 sparse-checkout set --no-cone /sam2/ /setup.py /README.md /LICENSE /MANIFEST.in
timeout 180s git -C /opt/sam2 fetch --depth=1 --filter=blob:none --no-tags --progress origin "${sam2_source##*#}"
timeout 180s git -C /opt/sam2 checkout --detach FETCH_HEAD
SH

RUN --mount=type=cache,id=nba-reconstruction-uv,target=/root/.cache/uv \
    uv sync --locked --extra gpu --no-install-project --no-install-package sam-2 --python 3.11

RUN git clone https://github.com/open-mmlab/mmpose.git /opt/mmpose \
    && git -C /opt/mmpose checkout --detach 759b39c13fea6ba094afc1fa932f51dc1b11cbf9 \
    && git clone https://github.com/Walter0807/MotionBERT.git /opt/MotionBERT \
    && git -C /opt/MotionBERT checkout --detach 705d3a95354db8bdb696b3492e47a3b5537174ff \
    && git config --system --add safe.directory /opt/MotionBERT

RUN --mount=type=cache,id=nba-reconstruction-uv,target=/root/.cache/uv \
    set -e; \
    uv venv --python 3.11 /opt/pose2d; \
    uv venv --python 3.11 /opt/pose3d; \
    for worker in /opt/pose2d/bin/python /opt/pose3d/bin/python; do \
      uv pip install --python "$worker" \
        --index-url https://download.pytorch.org/whl/cu121 \
        'torch==2.1.2+cu121' 'torchvision==0.16.2+cu121'; \
      uv pip install --python "$worker" 'numpy==1.26.4' 'opencv-python==4.10.0.84'; \
    done

RUN --mount=type=cache,id=nba-reconstruction-uv,target=/root/.cache/uv \
    uv pip install --python /opt/pose2d/bin/python \
      'pip==26.2.1' 'setuptools==60.2.0' 'wheel==0.48.0' \
    && uv pip install --python /opt/pose2d/bin/python \
      --no-build-isolation-package chumpy 'chumpy==0.70' \
    && uv pip install --python /opt/pose2d/bin/python \
      'mmengine==0.10.7' 'mmdet==3.2.0' \
      'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl' \
      -e /opt/mmpose \
    && uv pip install --python /opt/pose3d/bin/python 'easydict==1.13' 'PyYAML==6.0.3'

RUN mkdir -p /opt/weights \
    && curl -fL --retry 3 -o /opt/weights/rtmpose.pth \
      https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth \
    && curl -fL --retry 3 -o /opt/weights/motionbert.bin \
      https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin

COPY src ./src
# Keep the project editable so pose-demo shares the source tree with its workers.
RUN --mount=type=cache,id=nba-reconstruction-uv,target=/root/.cache/uv \
    uv sync --locked --extra gpu --no-install-package sam-2 --python 3.11 \
    && uv pip install --python /opt/tracking/bin/python --no-deps /opt/sam2

ENV PATH="/opt/tracking/bin:$PATH" \
    HOME=/tmp \
    HF_HOME=/model-cache/huggingface \
    MODEL_CACHE_DIR=/model-cache/roboflow

COPY --chmod=755 run-pipeline.sh /usr/local/bin/run-pipeline
ENTRYPOINT ["/usr/local/bin/run-pipeline"]
