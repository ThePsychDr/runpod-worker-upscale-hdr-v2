#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_models.sh — Download missing model weights
# Works locally and in Docker containers
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR="${MODEL_DIR:-/workspace/models}"

mkdir -p \
    "$MODEL_DIR/realesrgan" \
    "$MODEL_DIR/gfpgan" \
    "$MODEL_DIR/hdrtvdm"

# Returns 0 if file exists and is >1MB (not a partial download)
model_ok() {
    local f="$1"
    [[ -f "$f" ]] && [[ $(stat -c%s "$f" 2>/dev/null || echo 0) -gt 1048576 ]]
}

download() {
    local url="$1"
    local dest="$2"
    local name
    name=$(basename "$dest")

    if model_ok "$dest"; then
        local size
        size=$(du -sh "$dest" 2>/dev/null | cut -f1)
        echo "  [OK] $name ($size)"
        return 0
    fi

    echo "  [DL] $name..."
    if wget -q --show-progress -O "${dest}.tmp" "$url" 2>&1; then
        mv "${dest}.tmp" "$dest"
        local size
        size=$(du -sh "$dest" 2>/dev/null | cut -f1)
        echo "  [OK] $name ($size)"
    else
        rm -f "${dest}.tmp"
        echo "  [!!] FAILED: $name"
        return 1
    fi
}

echo "  Model directory: $MODEL_DIR"
echo ""

download \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth" \
    "$MODEL_DIR/realesrgan/realesr-general-x4v3.pth"

download \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth" \
    "$MODEL_DIR/realesrgan/realesr-general-wdn-x4v3.pth"

download \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
    "$MODEL_DIR/realesrgan/RealESRGAN_x4plus.pth"

download \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
    "$MODEL_DIR/gfpgan/GFPGANv1.4.pth"

download \
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" \
    "$MODEL_DIR/gfpgan/detection_Resnet50_Final.pth"

download \
    "https://github.com/AndreGuo/HDRTVDM/raw/main/method/params.pth" \
    "$MODEL_DIR/hdrtvdm/params.pth"

download \
    "https://github.com/AndreGuo/HDRTVDM/raw/main/method/params_3DM.pth" \
    "$MODEL_DIR/hdrtvdm/params_3DM.pth"

download \
    "https://github.com/AndreGuo/HDRTVDM/raw/main/method/params_DaVinci.pth" \
    "$MODEL_DIR/hdrtvdm/params_DaVinci.pth"

echo ""
echo "  All models checked."
