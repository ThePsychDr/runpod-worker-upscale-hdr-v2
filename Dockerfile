# ─────────────────────────────────────────────────────────────────────────────
# AI Upscale & HDR Pipeline — RunPod Serverless Worker
# ─────────────────────────────────────────────────────────────────────────────

FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204

LABEL maintainer="thepsych"
LABEL description="AI video upscaling pipeline — serverless worker"

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
        --enable-cuda-nvcc \
        --enable-cuvid \
        --enable-nvenc \
        --enable-nvdec \
        --enable-libnpp \
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

# ─── Python dependencies ──────────────────────────────────────────────────────

RUN pip install --no-cache-dir --no-deps torchvision==0.24.1 && \
    pip install --no-cache-dir \
    basicsr==1.4.2 \
    realesrgan==0.3.0 \
    gfpgan==1.3.8 \
    facexlib==0.3.0 \
    ffmpeg-python \
    opencv-python-headless \
    numpy \
    tqdm \
    scipy \
    runpod \
    boto3

# Fix basicsr: torchvision.transforms.functional_tensor removed in torchvision 0.19+
RUN python3 -c "import torchvision; print(f'torchvision {torchvision.__version__} installed')" && \
    SITE=$(python3 -c "import torchvision,os; print(os.path.dirname(torchvision.__file__))") && \
    mkdir -p "$SITE/transforms" && \
    echo "from torchvision.transforms.functional import *" > "$SITE/transforms/functional_tensor.py" && \
    echo "Created functional_tensor.py shim at $SITE/transforms/"

# Fix basicsr: _no_grad_trunc_normal_ import may be removed in newer torch
RUN find / -path "*/basicsr/archs/arch_util.py" -exec \
    python3 -c "import sys; p=sys.argv[1]; t=open(p).read(); \
    t=t.replace('from torch.nn.init import _no_grad_trunc_normal_', \
    'try:\\n    from torch.nn.init import _no_grad_trunc_normal_\\nexcept ImportError:\\n    from torch.nn.init import trunc_normal_ as _no_grad_trunc_normal_'); \
    open(p,'w').write(t)" {} \;

# Fix facexlib: pretrained= parameter deprecated in torchvision 0.13+
RUN find / -path "*/facexlib/detection/retinaface.py" -exec \
    sed -i 's/models\.resnet50(pretrained=False)/models.resnet50(weights=None)/' {} \;

ENV PYTHONWARNINGS="ignore::FutureWarning"

# ─── HDRTVDM ─────────────────────────────────────────────────────────────────

RUN pip install --no-cache-dir einops

RUN git clone https://github.com/AndreGuo/HDRTVDM /workspace/hdrtvdm && \
    cd /workspace/hdrtvdm && \
    pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

COPY src/models_hdrtvdm.py /workspace/hdrtvdm/models_hdrtvdm.py
ENV PYTHONPATH="/workspace/hdrtvdm"

# ─── Project files ────────────────────────────────────────────────────────────

WORKDIR /workspace

COPY src/upscale_hdr.py    /workspace/upscale_hdr.py
COPY src/grade_detect.py   /workspace/grade_detect.py
COPY src/handler.py        /workspace/handler.py

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
