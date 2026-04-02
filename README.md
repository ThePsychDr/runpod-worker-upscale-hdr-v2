# AI Video Upscale & HDR — RunPod Serverless Worker

[![Deploy on RunPod](https://badge.runpod.io/deploy.svg)](https://www.runpod.io/console/hub/ThePsychDr/runpod-ai-video-upscale-hdr)

Four-stage GPU pipeline for video enhancement:

1. **Denoise** — Real-ESRGAN artifact cleanup at source resolution
2. **Upscale** — RealESRGAN_x4plus adaptive upscale (up to 4K)
3. **Face Restore** — GFPGAN v1.4 facial detail recovery
4. **HDR Tone Map** — HDRTVDM SDR → HDR10/HDR10+ inverse tone mapping

All stages are optional. The pipeline auto-detects optimal settings based on input resolution, bitrate, and GPU VRAM.

## Performance

| GPU | ~sec/frame (4K output) |
|-----|----------------------|
| RTX 4090 | ~2.0s |
| L40S | ~2.2s |
| A6000 | ~3.0s |

## Quick Start

### Deploy

1. Click the **Deploy on RunPod** badge above, or search **"AI Video Upscale & HDR"** in the [RunPod Hub](https://www.runpod.io/console/hub)
2. Set your S3-compatible bucket credentials as environment variables (see [Configuration](#configuration))
3. Deploy

### Submit a job

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_url": "s3://your-bucket/input/video.mp4"
    }
  }'
```

### Python SDK

```python
import runpod

runpod.api_key = "YOUR_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT")

run = endpoint.run({"video_url": "s3://your-bucket/input/video.mp4"})
print(run.status())
output = run.output()
print(output["output_url"])
```

### Response

```json
{
  "output_url": "https://presigned-url-to-output.mkv",
  "filename": "video_hdr.mkv",
  "size_mb": 512.5
}
```

### Check status

```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Configuration

### Required: S3 bucket credentials

Set these as **environment variables** on your RunPod endpoint:

| Variable | Description |
|----------|-------------|
| `BUCKET_ENDPOINT_URL` | S3-compatible endpoint (e.g. `https://ACCOUNT.r2.cloudflarestorage.com`) |
| `BUCKET_ACCESS_KEY_ID` | Access key |
| `BUCKET_ACCESS_KEY_SECRET` | Secret key |
| `BUCKET_NAME` | Bucket name |

Input videos are read from the bucket via `video_url`. Output is uploaded to `output/JOB_ID/` in the same bucket.

### Optional: default overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_HDR_MODE` | `hdr10` | HDR output mode (`hdr10`, `hdr10plus`, `sdr`) |
| `DEFAULT_CRF` | `18` | Encoder quality (0–51, lower = better) |
| `DEFAULT_TARGET` | `0` (auto) | Force output height (e.g. `2160`) |
| `DEFAULT_DN` | `-1` (auto) | Denoise strength (0.0–1.0, -1 = auto from bitrate) |
| `DEFAULT_FACE_STRENGTH` | `0.5` | Face restoration blend (0.0 = aggressive, 1.0 = bypass) |
| `DEFAULT_NO_FACE` | `false` | Skip face restoration |
| `DEFAULT_PRESERVE_GRAIN` | `false` | Re-apply film grain after processing |
| `DEFAULT_PRESET` | `p7` | NVENC preset (p1–p7, higher = better quality) |

## Input Schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video_url` | string | **required** | S3/R2 key or URL to input video |
| `output_filename` | string | auto | Output filename |
| `target` | int | auto | Output height in pixels (e.g. `1440`, `2160`). Auto-selects based on source resolution and bitrate |
| `dn` | float | auto | Denoise strength in upscaler (0.0–1.0, -1 = auto from resolution + bitrate) |
| `no_denoise` | bool | false | Skip the first-stage artifact cleanup pass |
| `face_strength` | float | 0.5 | Face restoration blend (0.0 = aggressive, 1.0 = bypass) |
| `no_face` | bool | false | Skip face restoration |
| `no_itm` | bool | false | Output SDR — skip HDR tone mapping entirely |
| `hdr_mode` | string | hdr10 | HDR output mode (`hdr10`, `hdr10plus`) |
| `itm` | string | params_3DM | HDR checkpoint (`params_3DM`, `params_DaVinci`) |
| `crf` | int | 18 | Encoder quality (0–51, lower = better) |
| `preset` | string | p7 | Encoder preset (p1–p7, p7 = best quality) |
| `fp16` | bool | true | FP16 inference (faster, less VRAM) |
| `force_full_range` | bool | false | Treat input as full range (pc) |
| `preserve_grain` | bool | false | Re-apply film grain after processing |
| `grain_strength` | float | 1.0 | Grain intensity multiplier (0.0–2.0) |
| `highlight_boost` | float | 0.0 | Specular highlight expansion (0.0–1.0) |
| `temporal_smooth` | float | — | Temporal smoothing to reduce frame-to-frame flicker |
| `deinterlace` | string | auto | Deinterlace mode (`auto`/`on`/`off`) |
| `batch` | int | 4 | ITM batch size |
| `workers` | int | 16 | Decode threads |
| `start_time` | float | — | Start time in seconds (for chunked processing) |
| `chunk_duration` | float | — | Duration in seconds (for chunked processing) |

## Output Modes

### SDR (no HDR)

Skip tone mapping — output upscaled SDR. Useful when you want to grade HDR manually or the content doesn't need HDR.

```json
{ "video_url": "input/video.mp4", "no_itm": true }
```

### HDR10 (default)

Static metadata — single MaxCLL/MaxFALL values computed from all frames. Works on all HDR displays.

```json
{ "video_url": "input/video.mp4" }
```

Or explicitly:
```json
{ "video_url": "input/video.mp4", "hdr_mode": "hdr10" }
```

### HDR10+ (dynamic metadata)

Per-frame dynamic metadata with bezier curves for scene-by-scene tone mapping. Slower encode (libx265 with `--dhdr10-info`). Benefits Samsung HDR10+ displays (S95B, S95C, S95D, etc.). Falls back to HDR10 static on non-HDR10+ displays.

```json
{ "video_url": "input/video.mp4", "hdr_mode": "hdr10plus" }
```

### Encoding Details

| Mode | Encoder | Metadata | Compatibility |
|------|---------|----------|---------------|
| SDR | NVENC hevc_nvenc (GPU) or libx265 (CPU fallback) | None | All displays |
| HDR10 | libx265 10-bit | Static MaxCLL/MaxFALL + mastering display SEI | All HDR displays |
| HDR10+ | libx265 10-bit | Static + per-frame dynamic bezier curves | Samsung HDR10+ displays (falls back to HDR10 on others) |

HDR modes always use **libx265 software encode** because NVENC cannot inject HDR metadata into the HEVC bitstream. SDR mode uses NVENC when available for faster GPU-accelerated encoding.

## Pipeline Stages

| Stage | Model | Purpose |
|-------|-------|---------|
| 1 | realesr-general-x4v3 + WDN | Artifact cleanup at source resolution |
| 2 | RealESRGAN_x4plus | AI upscaling (adaptive, max 4K) |
| 3 | GFPGANv1.4 | Face restoration (auto-detected, skipped if no faces) |
| 4 | HDRTVDM (params_3DM) | SDR → HDR10 inverse tone mapping |

## GPU Compatibility

**Base image:** `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204` (PyTorch 2.9.1 + CUDA 12.8.1)
**Minimum:** CUDA compute capability sm_80+ with 24GB VRAM for full 4K pipeline.

### RunPod Pools

| Pool ID | GPUs | VRAM | Arch | Status |
|---------|------|------|------|--------|
| `ADA_24` | RTX 4090 | 24GB | Ada (sm_89) | **Proven** — primary test GPU, best cost/perf |
| `ADA_48_PRO` | L40, L40S, RTX 6000 Ada | 48GB | Ada (sm_89) | **Proven** (L40S) — no tiling needed |
| `AMPERE_24` | L4, A5000, RTX 3090 | 24GB | Ada/Ampere (sm_86–89) | **Compatible** — all 3 GPUs have NVENC/NVDEC, TF32 tensor cores, 24GB VRAM |
| `AMPERE_48` | A6000, A40 | 48GB | Ampere (sm_86) | **Compatible** — ~3s/frame, no tiling needed |
| `AMPERE_80` | A100 | 80GB | Ampere (sm_80) | Compatible — no NVENC (SDR encode falls back to libx265 CPU, HDR unaffected) |
| `ADA_80_PRO` | H100 | 80GB | Hopper (sm_90) | Compatible — overkill for this workload |
| `HOPPER_141` | H200 | 141GB | Hopper (sm_90) | Compatible — overkill |
| `AMPERE_16` | A4000, A4500, RTX 4000/2000 | 16GB | Ampere (sm_86) | Tight — heavy tiling required, max ~1440p practical |

### NOT supported on serverless (Blackwell)

RTX 5090, RTX 5080, B200, RTX PRO 6000 — Blackwell architecture (sm_100/sm_120).

**Why:** The serverless base image (`runpod/pytorch:1.0.3`) ships PyTorch 2.9.1 which was released before Blackwell GPUs existed. PyTorch 2.9.1 does not include sm_100 or sm_120 CUDA kernels, so any CUDA operation fails immediately on Blackwell hardware. This was tested — the worker crashes on startup when assigned a 5090.

**When will it work:** When RunPod updates their base image to PyTorch 2.11+ (which added Blackwell support). No code changes needed on our side — just a base image swap.

**Local workaround:** The local Dockerfile uses `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` which supports Blackwell. RTX 5090 works locally.

### Architecture Details

| Feature | Required | Used For |
|---------|----------|----------|
| CUDA sm_80+ | Yes | PyTorch 2.9.1 minimum compute capability |
| 24GB+ VRAM | Recommended | 4K output without tiling (16GB works with tiling up to ~1440p) |
| TF32 tensor cores | Recommended | ~20-30% matmul speedup (Ampere+ all have this) |
| NVDEC | Optional | h264_cuvid hardware decode — falls back to CPU if unavailable |
| NVENC | Optional | hevc_nvenc for SDR encode only — HDR always uses libx265 (CPU) |
| FP16 | Yes (default) | Half-precision inference for all 4 stages |

### ITM Inference Strategy

Stage 4 (HDR tone mapping) uses a 3-step VRAM fallback:

1. **Direct full-frame** — best quality, fastest. Works when VRAM fits the entire output frame
2. **Tiled inference** — same quality, overlapping 960px tiles with linear blend seams. Fits 4K on 24GB GPUs. ~15x faster than guide fallback
3. **Guide-based ratio transfer** — always works regardless of VRAM. Slightly less detail

### Recommended Endpoint GPU Priority

For best cost/performance, select these pools in order:

1. `ADA_24` (RTX 4090) — $0.00031/s, ~2.0s/frame
2. `ADA_48_PRO` (L40S/L40) — $0.00053/s, ~2.0s/frame
3. `AMPERE_24` (L4/A5000/3090) — $0.00019/s, ~2.5-3.5s/frame (est.)
4. `AMPERE_48` (A6000/A40) — $0.00034/s, ~3.0s/frame

## Model Licenses

| Model | License | Source |
|-------|---------|--------|
| Real-ESRGAN | BSD 3-Clause | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| GFPGAN | Apache 2.0 | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) |
| facexlib | MIT | [xinntao/facexlib](https://github.com/xinntao/facexlib) |
| basicsr | Apache 2.0 | [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) |
| HDRTVDM | Research use | [AndreGuo/HDRTVDM](https://github.com/AndreGuo/HDRTVDM) |

This worker code is released under the MIT License. Model weights are subject to their respective licenses.

## Changelog

| Version | Change |
|---------|--------|
| v1.7.4 | Add GitHub Actions workflow for Docker build & push to Docker Hub |
| v1.7.3 | Use `ADA_24` pool tier ID for Hub tests (replaces legacy GPU name) |
| v1.7.2 | Repo consolidation release — remove CACHEBUST ARG, revert test config to proven RTX 4090 settings, fix `no_denoise` flag, update docs |
| v1.7.1 | NVENC two-pass encode (last release on `runpod-video-upscale-hdr`) |
| v1.2.0 | Tiled ITM inference — 4K HDR on 24GB GPUs without OOM. ~15x faster than guide fallback |
| v1.1.0 | First stable release — all build fixes resolved, end-to-end tested on RunPod Hub |
| v1.0.0 | Initial release — 4-stage pipeline with baked models |
