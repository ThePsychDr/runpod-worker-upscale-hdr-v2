# Planned Features

## Implemented

### v1.7.10 — NVEncC GPU HDR10 encode

- **NVEncC baked into the image** — rigaya's standalone NVENC encoder (v9.14) is installed via the official `.deb` package and symlinked to `/usr/local/bin/NVEncC64`. Initial attempt built from source (`git clone --branch 7.85` + cmake + make) failed because `7.85` doesn't exist as a branch/tag on rigaya/NVEnc — the real tags are in the 9.xx series, and the upstream build script uses `./configure && make` on Linux rather than cmake. The `.deb` package avoids all of this complexity plus any libav* linkage conflicts with the from-source ffmpeg.
- **GPU HDR10 encode** — `upscale_hdr.py` now detects `NVEncC64` at import time via `_has_nvencc()` and routes HDR10 mode through `encode_hdr10_nvencc()` when available. NVEncC injects `--master-display` and `--max-cll` SEI metadata directly into the HEVC bitstream — something FFmpeg's `hevc_nvenc` can't do (the 2019 patches were never merged upstream)
- **Automatic fallback to libx265** — if NVEncC fails (bad exit code, missing binary, unsupported GPU like A100 which has no NVENC hardware), the HDR10 path falls back to the existing `encode_hdr10_static_twopass()` libx265 path. Fallback is transparent to the caller — same output file, same metadata, just slower
- **HDR10+ unchanged** — HDR10+ mode still uses libx265 with `--dhdr10-info` because NVEncC doesn't support per-frame dynamic metadata. Only HDR10 static mode benefits from the GPU encode path
- **Runtime check at worker startup** — the worker log now prints whether NVEncC is available and which binary it found
- **Performance:** GPU HDR10 encode is ~2-5× faster than libx265 on the same GPU, and frees up CPU threads for other pipeline stages. Full measurement TBD on 4090 vs L40S

### v1.7.8 — HDR Mastering Metadata Export

- **`export_metadata` flag** — writes HDR metadata sidecar files alongside the output for mastering workflows in DaVinci Resolve / manual x265 grading
  - `<name>_metadata.csv` — per-frame luminance stats (max_nits, avg_nits, percentiles p1/p5/p10/p25/p50/p75/p90/p95/p99) with MaxCLL/MaxFALL summary line
  - `<name>_hdr10plus.json` — HDR10+ dynamic metadata (Samsung spec), usable for both HDR10 static and HDR10+ dynamic grades
- **Handler integration** — `"export_metadata": true` in job input uploads both sidecar files to R2 and returns `metadata_csv_url` + `hdr10plus_json_url` in the response
- **HDR-only** — flag is a no-op in SDR/`no_itm` mode; warning printed on the worker
- **See `docs/DAVINCI_HDR_GRADING.md` (local repo)** for the full grading workflow reference

### v1.7.6 — x265 Threading + Output Modes

- **x265 threading optimization** — explicit `pools=N:wpp=1:lookahead-slices=4` on all libx265 encode paths. 1.5-3x faster HDR encoding with no quality impact
- **Output modes** — SDR, HDR10, HDR10+ documented with job input examples and encoding details
- **`hdr_mode: sdr`** — added `sdr` as valid API option (equivalent to `no_itm: true`)
- **GitHub Actions** — manual Docker build & push workflow for `thepsych/upscale-hdr:serverless`

### v1.7.x — Consolidation + GPU Compatibility

- **Repo consolidation** — 4 repos merged into 2 (this public repo + private local/TRT repo)
- **`ADA_24` pool ID** for Hub tests (replaces legacy `NVIDIA GeForce RTX 4090` name)
- **Removed CACHEBUST** — was causing infinite Hub rebuilds
- **GPU compatibility docs** — full pool mapping, Blackwell explanation, NVENC/NVDEC details
- **`no_denoise` flag** — skip first-stage artifact cleanup
- **Fixed deploy badge** — shields.io (RunPod badge API is down)
- **MaxCLL/MaxFALL** — replaced hardcoded values with `YOUR_MAXCLL`/`YOUR_MAXFALL` placeholders in docs

### v1.2.0 — HDR Quality + Stability

- **Tiled ITM inference** — 4K HDR on 24GB GPUs without OOM (~15x faster than guide fallback)
- **`highlight_boost`** — specular highlight expansion for HDR pop (0.0–1.0)
- **`temporal_smooth`** — reduce frame-to-frame flicker (default 0.15)
- **CUDA runtime API Docker build** — no nvcc required, smaller image

### v1.1.0 — Core Features

- **Manual chunk splitting** — `start_time` + `chunk_duration` for multi-GPU parallel processing
- **Per-video adaptive denoise** — `dn: -1` auto-detects from resolution + bitrate
- **Film grain preservation** — `preserve_grain` + `grain_strength`

## TensorRT Acceleration (private builds only)

TRT engines are GPU-architecture-specific — an engine built on Ada (4090) won't run on Hopper (H100) or Ampere (A40). Since serverless assigns arbitrary GPU types from a pool, TRT acceleration is not viable for public Hub releases.

Pre-built engines per architecture (sm_86, sm_89, sm_90) COPYed into the Docker image would solve this but is not yet implemented. TRT support is tested on fixed-GPU deployments (pods, local).

## Planned

### Per-Scene Adaptive Denoise (research)

Vary DN strength within a single video based on per-frame compression artifact analysis. Current `dn: -1` applies one value to the entire video — scenes with different compression quality (e.g. dark scenes with more blocking vs bright scenes with less) would benefit from per-frame adaptation.

**Approach under investigation:**
- Analyze each frame's compression artifact density before Stage 1 (variance of Laplacian for blur/blocking, DCT coefficient analysis for quantization artifacts)
- Map artifact severity to DN strength (heavy blocking → DN 0.7, clean frame → DN 0.15)
- Smooth DN values with EMA (like `TemporalSmoother`) to avoid jarring quality shifts at scene cuts
- Feasible because `run_stage1()` accepts DN per-call — it controls the blend weight between the general model and the WDN (weighted denoise) model, no model reload needed

### Blackwell Support (waiting on RunPod)

RTX 5090/5080 and B200 require PyTorch 2.11+ for sm_100/sm_120 CUDA kernels. Blocked until RunPod releases an official serverless base image with PyTorch 2.11+. Current latest official image is PyTorch 2.8. No code changes needed — just a base image swap.
