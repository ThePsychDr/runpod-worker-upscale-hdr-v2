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
| `no_itm` | bool | false | Skip HDR tone mapping, output SDR |
| `itm` | string | params_3DM | HDR checkpoint (`params_3DM`, `params_DaVinci`) |
| `hdr_mode` | string | hdr10 | HDR output mode (`hdr10`, `hdr10plus`) |
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

## Pipeline Stages

| Stage | Model | Purpose |
|-------|-------|---------|
| 1 | realesr-general-x4v3 + WDN | Artifact cleanup at source resolution |
| 2 | RealESRGAN_x4plus | AI upscaling (adaptive, max 4K) |
| 3 | GFPGANv1.4 | Face restoration (auto-detected, skipped if no faces) |
| 4 | HDRTVDM (params_3DM) | SDR → HDR10 inverse tone mapping |

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
| v1.2.3 | Fix `no_denoise` flag not being passed to pipeline CLI |
| v1.2.2 | Fix Docker build (use CUDA runtime API, no nvcc required) + fix Hub test GPU availability |
| v1.2.1 | Worker version bump to v4.1 |
| v1.2.0 | Tiled ITM inference — 4K HDR on 24GB GPUs (RTX 4090, A5000, A6000) without OOM. ~15x faster than guide fallback |
| v1.1.0 | First stable release — all build fixes resolved, end-to-end tested on RunPod Hub |
| v1.0.x | Build iteration — torchvision install ordering, CUDA 12.8 compat, serverless endpoint fix |
| v1.0.0 | Initial release — 4-stage pipeline with baked models |
