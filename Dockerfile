# ─────────────────────────────────────────────────────────────────────────────
# AI Upscale & HDR Pipeline — RunPod Serverless Worker
# ─────────────────────────────────────────────────────────────────────────────

FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204

LABEL maintainer="thepsych"
LABEL description="AI video upscaling pipeline — serverless worker"

# ─── Fix python3 symlink ─────────────────────────────────────────────────────
# Base image has Python 3.12 (with torch) but python3 resolves to system 3.10.
# All pip installs go to 3.12, so python3 must point there too.
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    python3 --version

# ─── System dependencies ─────────────────────────────────────────────────────

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    wget \
    xz-utils \
    nasm \
    yasm \
    pkg-config \
    libx264-dev \
    libx265-dev \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── FFmpeg with NVDEC/NVENC (CUDA hardware decode + encode) ─────────────────

RUN git clone --branch n12.1.14.0 --depth 1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git /tmp/nv-codec-headers && \
    cd /tmp/nv-codec-headers && make install && rm -rf /tmp/nv-codec-headers

RUN git clone --branch n7.1 --depth 1 https://git.ffmpeg.org/ffmpeg.git /tmp/ffmpeg-src && \
    cd /tmp/ffmpeg-src && \
    PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure \
        --enable-gpl \
        --enable-nonfree \
        --enable-cuda \
        --enable-cuvid \
        --enable-nvenc \
        --enable-nvdec \
        --enable-libx264 \
        --enable-libx265 \
        --extra-cflags="-I/usr/local/cuda/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64" \
        --disable-doc \
        --disable-debug \
        --disable-static \
        --enable-shared && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/ffmpeg-src

# ─── NVEncC (rigaya) — GPU HDR10 encoder with SEI metadata injection ─────────
# NVEncC can inject --master-display and --max-cll SEI into the HEVC bitstream,
# unlike FFmpeg's hevc_nvenc which does not expose those options. Used for GPU-
# accelerated HDR10 encoding on any RunPod GPU with NVENC hardware (Ada, Ampere
# consumer, Hopper with NVENC). At runtime, upscale_hdr.py auto-detects the
# NVEncC64 binary on PATH and falls back to libx265 (CPU) if it's missing or
# the encode fails. A100 has no NVENC hardware, so it'll always use the libx265
# fallback there — HDR10+ mode always uses libx265 because of the dhdr10-info
# requirement.
#
# Installed from rigaya's official .deb package. Verified via deb inspection:
#   - Binary:  /usr/bin/nvencc
#   - Depends: libc6 (>=2.31)  — already satisfied by ubuntu2204 base
# So no `apt-get install -f` fallback needed.
#
# Runtime note: nvencc is a CUDA application and dynamically links against
# libcuda.so.1, which is NOT present during Docker build (the stub library
# lives on the host NVIDIA driver install and is mounted into the container
# by nvidia-container-toolkit when `--gpus all` is passed at runtime). We
# therefore can't run `NVEncC64 --version` as a build smoke test here —
# it would fail with "error while loading shared libraries: libcuda.so.1".
# Runtime verification happens in handler.py at worker startup where the
# driver is available and nvidia-container-toolkit has already mounted
# libcuda.so.1. Symlink /usr/bin/nvencc -> /usr/local/bin/NVEncC64 so the
# upscale_hdr.py _has_nvencc() probe and handler.py startup log find it
# under the expected name.
RUN wget -q -O /tmp/nvencc.deb \
    https://github.com/rigaya/NVEnc/releases/download/9.14/nvencc_9.14_amd64.deb && \
    dpkg -i /tmp/nvencc.deb && \
    rm /tmp/nvencc.deb && \
    ln -sf /usr/bin/nvencc /usr/local/bin/NVEncC64 && \
    test -x /usr/local/bin/NVEncC64 && \
    echo "NVEncC64 installed at /usr/local/bin/NVEncC64 (runtime --version check in handler.py)"

# ─── Python dependencies ──────────────────────────────────────────────────────

# Install AI packages with --no-deps to prevent pip from upgrading torch/torchvision
# to non-CUDA PyPI versions. Then install their non-torch dependencies separately.
RUN pip install --no-cache-dir --no-deps \
    basicsr==1.4.2 \
    realesrgan==0.3.0 \
    gfpgan==1.3.8 \
    facexlib==0.3.0 && \
    pip install --no-cache-dir \
    ffmpeg-python \
    opencv-python-headless \
    numpy \
    tqdm \
    scipy \
    lmdb \
    addict \
    yapf \
    Pillow \
    scikit-image \
    runpod \
    boto3

ENV PYTHONWARNINGS="ignore::FutureWarning"

# ─── HDRTVDM ─────────────────────────────────────────────────────────────────

RUN pip install --no-cache-dir einops

RUN git clone https://github.com/AndreGuo/HDRTVDM /workspace/hdrtvdm

COPY src/models_hdrtvdm.py /workspace/hdrtvdm/models_hdrtvdm.py
ENV PYTHONPATH="/workspace/hdrtvdm"

# ─── torchvision (MUST be last pip install) ──────────────────────────────────
# torch 2.9.1+cu128 is pre-installed in the base image.
# Install torchvision AFTER every other pip step so nothing can evict it.
# --no-deps prevents pip from pulling a CPU-only torch alongside it.
RUN pip install --no-cache-dir --no-deps torchvision==0.24.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Verify torchvision + create functional_tensor shim (separate RUN to flush pip caches)
RUN python3 -c "import torchvision; print(f'torchvision {torchvision.__version__} installed')" && \
    SITE=$(python3 -c "import torchvision,os; print(os.path.dirname(torchvision.__file__))") && \
    mkdir -p "$SITE/transforms" && \
    echo "from torchvision.transforms.functional import *" > "$SITE/transforms/functional_tensor.py" && \
    echo "Created functional_tensor.py shim at $SITE/transforms/"

# ─── Patch dependencies (must run AFTER torchvision is installed) ────────────

# Fix basicsr: _no_grad_trunc_normal_ removed in newer torch
RUN python3 -c "\
import basicsr.archs.arch_util as m; import os; p=m.__file__; \
t=open(p).read(); \
t=t.replace('from torch.nn.init import _no_grad_trunc_normal_', \
'try:\n    from torch.nn.init import _no_grad_trunc_normal_\nexcept ImportError:\n    from torch.nn.init import trunc_normal_ as _no_grad_trunc_normal_'); \
open(p,'w').write(t); print(f'Patched {p}')"

# Fix facexlib: pretrained= parameter deprecated in torchvision 0.13+
RUN python3 -c "\
import facexlib.detection.retinaface as m; import os; p=m.__file__; \
t=open(p).read(); \
t=t.replace('models.resnet50(pretrained=False)', 'models.resnet50(weights=None)'); \
open(p,'w').write(t); print(f'Patched {p}')"

# ─── Project files ────────────────────────────────────────────────────────────

WORKDIR /workspace

COPY src/upscale_hdr.py    /workspace/upscale_hdr.py
COPY src/grade_detect.py   /workspace/grade_detect.py
COPY handler.py            /workspace/handler.py

# ─── Bake models into image (eliminates cold start download) ──────────────────

RUN mkdir -p \
    /workspace/models/realesrgan \
    /workspace/models/gfpgan \
    /workspace/models/hdrtvdm \
    /workspace/gfpgan/weights \
    /tmp/output

RUN wget -q -O /workspace/models/realesrgan/realesr-general-x4v3.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth" && \
    wget -q -O /workspace/models/realesrgan/realesr-general-wdn-x4v3.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth" && \
    wget -q -O /workspace/models/realesrgan/RealESRGAN_x4plus.pth \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" && \
    wget -q -O /workspace/models/gfpgan/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" && \
    wget -q -O /workspace/models/gfpgan/detection_Resnet50_Final.pth \
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" && \
    cp /workspace/models/gfpgan/detection_Resnet50_Final.pth \
    /workspace/gfpgan/weights/detection_Resnet50_Final.pth && \
    wget -q -O /workspace/models/hdrtvdm/params.pth \
    "https://github.com/AndreGuo/HDRTVDM/raw/main/method/params.pth" && \
    wget -q -O /workspace/models/hdrtvdm/params_3DM.pth \
    "https://github.com/AndreGuo/HDRTVDM/raw/main/method/params_3DM.pth" && \
    wget -q -O /workspace/models/hdrtvdm/params_DaVinci.pth \
    "https://github.com/AndreGuo/HDRTVDM/raw/main/method/params_DaVinci.pth"

COPY src/download_models.sh /workspace/download_models.sh
RUN chmod +x /workspace/download_models.sh

# ─── Serverless entry point ──────────────────────────────────────────────────

CMD ["python", "-u", "/workspace/handler.py"]
