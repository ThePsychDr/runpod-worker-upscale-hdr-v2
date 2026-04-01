# Planned Features

## Implemented

### v1.2.0–v1.2.3 — HDR Quality + Stability

- **Tiled ITM inference** — 4K HDR on 24GB GPUs without OOM (~15x faster than guide fallback)
- **`highlight_boost`** — specular highlight expansion for HDR pop (0.0–1.0)
- **`temporal_smooth`** — reduce frame-to-frame flicker (default 0.4)
- **`no_denoise` flag** — skip first-stage artifact cleanup
- **CUDA runtime API Docker build** — no nvcc required, smaller image

## TensorRT Acceleration (private builds only)

TRT engines are GPU-architecture-specific — an engine built on Ada (4090) won't run on Hopper (H100) or Ampere (A40). Since serverless assigns arbitrary GPU types, TRT acceleration is not viable for public Hub releases. TRT support is being tested on fixed-GPU deployments (pods, local).

## Implemented

### Manual Chunk Splitting

`start_time` and `chunk_duration` parameters allow processing a segment of the video. Split long videos across multiple workers by submitting parallel jobs:

```bash
# Split a 5-minute video into 5 × 60s chunks across 5 workers
for i in 0 60 120 180 240; do
  curl -s https://api.runpod.ai/v2/YOUR_ENDPOINT/run \
    -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" \
    -d "{\"input\": {
      \"video_url\": \"input/video.mp4\",
      \"start_time\": $i,
      \"chunk_duration\": 60,
      \"target\": 2160
    }}"
done

# After all jobs complete, concatenate locally:
# 1. Create file list
for f in chunk_*.mkv; do echo "file '$f'" >> list.txt; done
# 2. Lossless concat (no re-encode)
ffmpeg -f concat -safe 0 -i list.txt -c copy final.mkv
```

### Per-Video Adaptive Denoise

`dn: -1` auto-detects optimal denoise strength from source resolution + bitrate via `recommend_dn()`. Applied uniformly to the entire video. Threshold table covers 480p through 1080p+ at various bitrate ranges.

## Planned

### Per-Scene Adaptive Denoise (research)

Vary DN strength within a single video based on per-frame compression artifact analysis. Current `dn: -1` applies one value to the entire video — scenes with different compression quality (e.g. dark scenes with more blocking vs bright scenes with less) would benefit from per-frame adaptation.

**Approach under investigation:**
- Analyze each frame's compression artifact density before Stage 1 (variance of Laplacian for blur/blocking, DCT coefficient analysis for quantization artifacts)
- Map artifact severity to DN strength (heavy blocking → DN 0.7, clean frame → DN 0.15)
- Smooth DN values with EMA (like `TemporalSmoother`) to avoid jarring quality shifts at scene cuts
- Feasible because `run_stage1()` accepts DN per-call — it controls the blend weight between the general model and the WDN (weighted denoise) model, no model reload needed
