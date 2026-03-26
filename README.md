# AI Video Upscale & HDR — RunPod Serverless Worker

Four-stage GPU pipeline for video enhancement:

1. **Denoise** — Real-ESRGAN artifact cleanup at source resolution
2. **Upscale** — RealESRGAN_x4plus adaptive upscale (up to 4K)
3. **Face Restore** — GFPGAN v1.4 facial detail recovery
4. **HDR Tone Map** — HDRTVDM SDR to HDR10/HDR10+ inverse tone mapping

All stages are optional. The pipeline auto-detects optimal settings based on input resolution, bitrate, and GPU VRAM.

## Performance

| GPU | ~sec/frame (4K output) | Cost/1000 frames |
|-----|----------------------|-----------------|
| RTX 4090 | ~2.0s | $0.62 |
| L40S | ~2.2s | $1.17 |
| A6000 | ~3.0s | $0.96 |

## Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Search for "AI Video Upscale & HDR" in the Hub
3. Configure your S3-compatible bucket credentials (Cloudflare R2, AWS S3, etc.)
4. Deploy

## API

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

### Full input schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video_url` | string | **required** | S3/R2 URL to input video |
| `output_filename` | string | auto | Output filename |
| `dn` | float | -1 (auto) | Denoise strength (0.0-1.0, -1 = auto from bitrate) |
| `face_strength` | float | 0.5 | Face restoration blend (0.0 = aggressive, 1.0 = bypass) |
| `no_face` | bool | false | Skip face restoration |
| `no_itm` | bool | false | Skip HDR, output SDR |
| `itm` | string | params_3DM | HDR checkpoint (params_3DM, params_DaVinci) |
| `hdr_mode` | string | hdr10 | HDR output mode (hdr10, hdr10plus) |
| `crf` | int | 18 | Encoder quality (0-51, lower = better) |
| `preset` | string | p7 | Encoder preset (p1-p7, p7 = best quality) |
| `fp16` | bool | true | FP16 inference (faster, less VRAM) |
| `target` | int | 0 (auto) | Force output height (e.g. 2160) |
| `force_full_range` | bool | false | Treat input as full range (pc) |
| `preserve_grain` | bool | false | Re-apply film grain after processing |
| `grain_strength` | float | 1.0 | Grain intensity multiplier (0.0-2.0) |
| `deinterlace` | string | auto | Deinterlace mode (auto/on/off) |
| `batch` | int | 4 | ITM batch size |
| `workers` | int | 16 | Decode threads |
| `start_time` | float | - | Start time in seconds (for chunked processing) |
| `chunk_duration` | float | - | Duration in seconds (for chunked processing) |

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

## Requirements

- **S3-compatible bucket** for video I/O (Cloudflare R2, AWS S3, MinIO, etc.)
- Upload input videos to the bucket, pass the S3 URL as `video_url`
- Output is uploaded back to the same bucket under `output/JOB_ID/`

## Pipeline stages

| Stage | Model | Purpose |
|-------|-------|---------|
| 1 | realesr-general-x4v3 + WDN | Artifact cleanup at source resolution |
| 2 | RealESRGAN_x4plus | AI upscaling (adaptive, max 4K) |
| 3 | GFPGANv1.4 | Face restoration (auto-detected, skipped if no faces) |
| 4 | HDRTVDM (params_3DM) | SDR to HDR10 inverse tone mapping |

## Model licenses

| Model | License | Source |
|-------|---------|--------|
| Real-ESRGAN | BSD 3-Clause | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) |
| GFPGAN | Apache 2.0 | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) |
| facexlib | MIT | [xinntao/facexlib](https://github.com/xinntao/facexlib) |
| basicsr | Apache 2.0 | [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) |
| HDRTVDM | Research use | [AndreGuo/HDRTVDM](https://github.com/AndreGuo/HDRTVDM) |

This worker code is released under the MIT License. Model weights are subject to their respective licenses.
