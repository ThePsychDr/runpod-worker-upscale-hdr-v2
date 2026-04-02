"""
AI Upscale & HDR Pipeline — Serverless Gradio UI
Submits jobs to RunPod serverless endpoint. Runs locally, no GPU needed.

Setup:
    pip install gradio requests boto3
    cp .env.example .env  # edit with your credentials
    python ui.py

Requires: ffmpeg, ffprobe, rclone (for R2/S3 file operations)
"""

import os
import re
import time
import json
import math
import threading
import subprocess
from pathlib import Path
from datetime import datetime

import gradio as gr
import requests

# ─── Load .env file if present ───────────────────────────────────────────────
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# ─── Config ──────────────────────────────────────────────────────────────────

def _get_api_key():
    config = Path.home() / ".runpod" / "config.toml"
    if config.exists():
        for line in config.read_text().splitlines():
            if line.strip().lower().startswith("apikey"):
                return line.split("=", 1)[1].strip().strip("'\"")
    return os.environ.get("RUNPOD_API_KEY", "")

def _detect_endpoints(api_key):
    try:
        r = requests.get("https://rest.runpod.io/v1/endpoints",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        return r.json()
    except Exception:
        return []

def _query_gpu_availability(api_key):
    """Query RunPod GraphQL for real-time GPU availability."""
    if not api_key:
        return {}
    try:
        r = requests.post(
            f"https://api.runpod.io/graphql?api_key={api_key}",
            json={"query": "{ gpuTypes { id displayName memoryInGb secureCloud communityCloud } }"},
            timeout=10)
        data = r.json().get("data", {}).get("gpuTypes", [])
        avail = {}
        for g in data:
            available = g.get("secureCloud", False) or g.get("communityCloud", False)
            avail[g["id"]] = {
                "name": g["displayName"],
                "vram": g.get("memoryInGb", 0),
                "available": available,
                "secure": g.get("secureCloud", False),
                "community": g.get("communityCloud", False),
            }
        return avail
    except Exception:
        return {}

API_KEY = _get_api_key()
ALL_ENDPOINTS = _detect_endpoints(API_KEY)
GPU_AVAILABILITY = _query_gpu_availability(API_KEY)

BUCKET_NAME = os.environ.get("BUCKET_NAME", "")
BUCKET_ENDPOINT_URL = os.environ.get("BUCKET_ENDPOINT_URL", "")
BUCKET_ACCESS_KEY_ID = os.environ.get("BUCKET_ACCESS_KEY_ID", "")
BUCKET_ACCESS_KEY_SECRET = os.environ.get("BUCKET_ACCESS_KEY_SECRET", "")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── GPU performance / cost lookup ───────────────────────────────────────────
# seconds per frame (full pipeline, no TensorRT) and RunPod serverless $/s
GPU_SPECS = {
    "NVIDIA GeForce RTX 5090":          {"spf": 1.6,  "cost_s": 0.00044, "vram": 32, "short": "RTX 5090"},
    "NVIDIA GeForce RTX 4090":          {"spf": 2.2,  "cost_s": 0.00031, "vram": 24, "short": "RTX 4090"},
    "NVIDIA L40S":                      {"spf": 2.2,  "cost_s": 0.00053, "vram": 48, "short": "L40S"},
    "NVIDIA L40":                       {"spf": 2.4,  "cost_s": 0.00034, "vram": 48, "short": "L40"},
    "NVIDIA RTX 6000 Ada Generation":   {"spf": 2.3,  "cost_s": 0.00067, "vram": 48, "short": "RTX 6000 Ada"},
    "NVIDIA RTX A6000":                 {"spf": 3.0,  "cost_s": 0.00032, "vram": 48, "short": "A6000"},
    "NVIDIA RTX A5000":                 {"spf": 15.0, "cost_s": 0.00026, "vram": 24, "short": "A5000"},
    "NVIDIA L4":                        {"spf": 5.4,  "cost_s": 0.00021, "vram": 24, "short": "L4"},
    "NVIDIA RTX 4080":                  {"spf": 3.5,  "cost_s": 0.00037, "vram": 16, "short": "RTX 4080"},
    "NVIDIA RTX A4000":                 {"spf": 8.0,  "cost_s": 0.00018, "vram": 16, "short": "A4000"},
}

# Startup overhead: model init + TRT engine build on first run (seconds)
# Models are baked into the Docker image (no download needed)
# TRT engine build adds ~60s on first job per worker, cached after
STARTUP_OVERHEAD_S = 15  # warm start with baked models + FlashBoot

# Map endpoint id → metadata from API
ENDPOINT_META = {}  # eid -> {name, gpuTypeIds, workersMax, ...}

# Pick first endpoint with "upscale" in name, else first available
DEFAULT_ENDPOINT = ""
ENDPOINT_CHOICES = []
for ep in ALL_ENDPOINTS:
    eid = ep.get("id", "")
    ename = ep.get("name", eid)
    ENDPOINT_CHOICES.append(f"{ename} ({eid})")
    ENDPOINT_META[eid] = ep
    if not DEFAULT_ENDPOINT and "upscale" in ename.lower():
        DEFAULT_ENDPOINT = f"{ename} ({eid})"
if ENDPOINT_CHOICES and not DEFAULT_ENDPOINT:
    DEFAULT_ENDPOINT = ENDPOINT_CHOICES[0]

def _extract_endpoint_id(choice):
    if not choice:
        return ""
    m = re.search(r'\(([^)]+)\)$', choice)
    return m.group(1) if m else choice

def _get_endpoint_gpu_specs(endpoint_choice):
    """Return (best_spec, worst_spec, gpu_list) for the selected endpoint."""
    eid = _extract_endpoint_id(endpoint_choice)
    meta = ENDPOINT_META.get(eid, {})
    gpu_ids = meta.get("gpuTypeIds", [])
    if not gpu_ids:
        default = {"spf": 2.2, "cost_s": 0.00044, "vram": 24, "short": "Unknown"}
        return default, default, []

    specs = []
    for g in gpu_ids:
        s = GPU_SPECS.get(g)
        if s:
            specs.append((g, s))
        else:
            # Unknown GPU — conservative estimate
            specs.append((g, {"spf": 4.0, "cost_s": 0.00040, "vram": 24,
                              "short": g.replace("NVIDIA ", "").replace("GeForce ", "")}))

    best = min(specs, key=lambda x: x[1]["spf"])
    worst = max(specs, key=lambda x: x[1]["spf"])
    return best[1], worst[1], specs

def _get_endpoint_workers_max(endpoint_choice):
    eid = _extract_endpoint_id(endpoint_choice)
    meta = ENDPOINT_META.get(eid, {})
    return meta.get("workersMax", 1)

def _get_endpoint_info_md(endpoint_choice):
    """Rich markdown summary of endpoint config — GPUs, costs, workers."""
    eid = _extract_endpoint_id(endpoint_choice)
    meta = ENDPOINT_META.get(eid, {})
    if not meta:
        return "No endpoint data."

    gpu_ids = meta.get("gpuTypeIds", [])
    wmax = meta.get("workersMax", 1)
    wmin = meta.get("workersMin", 0)
    wstandby = meta.get("workersStandby", 0)
    idle_s = meta.get("idleTimeout", 0)
    exec_ms = meta.get("executionTimeoutMs", 0)
    flashboot = meta.get("flashboot", False)
    scaler = meta.get("scalerType", "?")
    scaler_val = meta.get("scalerValue", "?")

    lines = [f"### Endpoint: `{meta.get('name', eid)}`"]
    lines.append("")

    # GPU table with availability
    lines.append("| GPU | Speed | Cost/s | Cost/hr | VRAM | Status |")
    lines.append("|-----|-------|--------|---------|------|--------|")
    for g in gpu_ids:
        s = GPU_SPECS.get(g, {"spf": "?", "cost_s": "?", "vram": "?", "short": g.replace("NVIDIA ", "")})
        short = s.get("short", g)
        spf = s.get("spf", "?")
        cs = s.get("cost_s", "?")
        vram = s.get("vram", "?")
        cost_hr = f"${cs * 3600:.2f}" if isinstance(cs, (int, float)) else "?"
        spf_str = f"{spf}s/frame" if isinstance(spf, (int, float)) else "?"
        cs_str = f"${cs:.5f}" if isinstance(cs, (int, float)) else "?"
        # Availability from GPU_AVAILABILITY
        ga = GPU_AVAILABILITY.get(g, {})
        if ga:
            if ga.get("available"):
                status = "Available"
            else:
                status = "Unavailable"
        else:
            status = "?"
        lines.append(f"| {short} | {spf_str} | {cs_str} | {cost_hr} | {vram}GB | {status} |")

    lines.append("")
    lines.append(f"| Setting | Value |")
    lines.append(f"|---------|-------|")
    lines.append(f"| Workers | {wmin} min / {wmax} max / {wstandby} standby |")
    lines.append(f"| Idle timeout | {idle_s}s |")
    lines.append(f"| Exec timeout | {_fmt_duration(exec_ms / 1000) if exec_ms else 'off'} |")
    lines.append(f"| FlashBoot | {'✅' if flashboot else '❌'} |")
    lines.append(f"| Scaler | {scaler} ({scaler_val}) |")

    return "\n".join(lines)

def refresh_gpu_availability(endpoint_choice):
    """Re-query RunPod for GPU availability and update the endpoint info."""
    global GPU_AVAILABILITY
    GPU_AVAILABILITY = _query_gpu_availability(API_KEY)
    return _get_endpoint_info_md(endpoint_choice)

def on_endpoint_change(endpoint_choice):
    """Update split slider and endpoint info when endpoint changes."""
    wmax = max(_get_endpoint_workers_max(endpoint_choice), 1)
    info_md = _get_endpoint_info_md(endpoint_choice)
    return (
        gr.update(value=wmax, maximum=max(wmax, 8)),
        info_md,
    )

HEADERS_BASE = {"Content-Type": "application/json"}

def _headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def _api_base(endpoint_id):
    return f"https://api.runpod.ai/v2/{endpoint_id}"

# ─── S3 Upload via rclone ────────────────────────────────────────────────────

def _sanitize_s3_key(name):
    return re.sub(r'[|&;<>]', '_', name)

def _r2_env():
    return {
        **os.environ,
        "RCLONE_CONFIG_S3_TYPE": "s3",
        "RCLONE_CONFIG_S3_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_S3_ENDPOINT": BUCKET_ENDPOINT_URL,
        "RCLONE_CONFIG_S3_ACCESS_KEY_ID": BUCKET_ACCESS_KEY_ID,
        "RCLONE_CONFIG_S3_SECRET_ACCESS_KEY": BUCKET_ACCESS_KEY_SECRET,
        "RCLONE_CONFIG_S3_ACL": "private",
        "RCLONE_CONFIG_S3_NO_CHECK_BUCKET": "true",
    }

def _r2_file_exists(s3_key):
    """Check if a file already exists in R2 with the same size."""
    result = subprocess.run(
        ["rclone", "lsf", "--format", "s", f"s3:{BUCKET_NAME}/{s3_key}"],
        env=_r2_env(), capture_output=True, text=True, timeout=15)
    return bool(result.stdout.strip())


def upload_to_r2(filepath):
    """Upload file to R2, return the S3 URL. Skip if already exists with same size."""
    safe_name = _sanitize_s3_key(os.path.basename(filepath))
    s3_key = f"input/{safe_name}"
    local_size = os.path.getsize(filepath)
    # Check if already in R2 with matching size
    result = subprocess.run(
        ["rclone", "lsf", "--format", "sp", f"s3:{BUCKET_NAME}/{s3_key}"],
        env=_r2_env(), capture_output=True, text=True, timeout=15)
    if result.stdout.strip():
        try:
            remote_size = int(result.stdout.strip().split(";")[0])
            if remote_size == local_size:
                return f"{BUCKET_ENDPOINT_URL}/{BUCKET_NAME}/{s3_key}"
        except (ValueError, IndexError):
            pass
    subprocess.run(["rclone", "copyto", filepath, f"s3:{BUCKET_NAME}/{s3_key}"],
                   env=_r2_env(), capture_output=True, timeout=600)
    return f"{BUCKET_ENDPOINT_URL}/{BUCKET_NAME}/{s3_key}"

def download_from_r2(s3_key, local_path):
    """Download file from R2."""
    subprocess.run(["rclone", "copyto", f"s3:{BUCKET_NAME}/{s3_key}", str(local_path)],
                   env=_r2_env(), capture_output=True, timeout=600)

# ─── Video Probing ───────────────────────────────────────────────────────────

def _probe_file(filepath):
    """Probe a file and return parsed dict with video stats."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(filepath)],
        capture_output=True, text=True, timeout=15)
    data = json.loads(result.stdout)
    info = {}
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            info["width"] = int(s.get("width", 0))
            info["height"] = int(s.get("height", 0))
            info["codec"] = s.get("codec_name", "?")
            info["pix_fmt"] = s.get("pix_fmt", "?")
            info["color_range"] = s.get("color_range")
            info["color_transfer"] = s.get("color_transfer")
            info["color_primaries"] = s.get("color_primaries")
            # Parse FPS — prefer r_frame_rate but fall back to avg_frame_rate
            # for VFR/mjpeg containers where r_frame_rate is the timebase (e.g. 90000/1)
            fps = 30
            for fps_key in ("r_frame_rate", "avg_frame_rate"):
                try:
                    num, den = s.get(fps_key, "0/1").split("/")
                    candidate = int(num) / int(den) if int(den) > 0 else 0
                    if 1 <= candidate <= 240:  # sane range
                        fps = candidate
                        break
                except (ValueError, ZeroDivisionError):
                    continue
            info["fps"] = fps
            info["fps_str"] = s.get("avg_frame_rate", s.get("r_frame_rate", "?"))
            # Use nb_frames if available (most accurate)
            nb = s.get("nb_frames")
            if nb and nb != "N/A":
                try:
                    info["nb_frames"] = int(nb)
                except ValueError:
                    pass
            break
    fmt = data.get("format", {})
    info["duration"] = float(fmt.get("duration", 0))
    br = fmt.get("bit_rate")
    info["bitrate_kbps"] = int(br) // 1000 if br else 0
    info["size_bytes"] = os.path.getsize(filepath)
    info["size_mb"] = info["size_bytes"] / 1048576
    # Use nb_frames from stream if available, otherwise calculate from duration × fps
    if "nb_frames" in info:
        info["total_frames"] = info["nb_frames"]
    else:
        info["total_frames"] = int(info["duration"] * info.get("fps", 30))
    return info

def _format_video_stats(info):
    """Format probe info into a nice stats block."""
    dur = info.get("duration", 0)
    mins, secs = divmod(dur, 60)
    cr = info.get("color_range") or "not signaled"
    ct = info.get("color_transfer") or "not signaled"
    is_hdr = ct in ("smpte2084", "arib-std-b67", "bt2020-10", "bt2020-12")
    lines = [
        f"**Resolution:** {info.get('width', '?')}x{info.get('height', '?')}",
        f"**Codec:** {info.get('codec', '?')}  |  **Pix fmt:** {info.get('pix_fmt', '?')}",
        f"**FPS:** {info.get('fps_str', '?')}  |  **Duration:** {int(mins)}m {secs:.1f}s",
        f"**Bitrate:** {info.get('bitrate_kbps', 0):,} kbps  |  **Size:** {info.get('size_mb', 0):.1f} MB",
        f"**Total frames:** {info.get('total_frames', 0):,}",
        f"**Color range:** {cr}  |  **Transfer:** {ct}",
    ]
    if is_hdr:
        lines.append("**Source is HDR** — ITM stage should be skipped (use Skip HDR checkbox)")
    return "\n".join(lines)

def _compact_stats(info):
    """Single-line compact stats."""
    dur = info.get("duration", 0)
    mins, secs = divmod(dur, 60)
    parts = [
        f"{info.get('width', '?')}x{info.get('height', '?')}",
        info.get("codec", "?"),
        f"{info.get('fps_str', '?')} fps",
        info.get("pix_fmt", "?"),
        f"{int(mins)}m{secs:.1f}s",
        f"{info.get('bitrate_kbps', 0)} kbps",
        f"{info.get('size_mb', 0):.1f} MB",
    ]
    return " | ".join(parts)

def probe_video(filepath, endpoint_choice="", split_gpus=1, no_itm=False):
    """Probe a video file and return formatted info with endpoint-aware estimates."""
    if not filepath or not os.path.exists(filepath):
        return "Select a video file first."
    try:
        info = _probe_file(filepath)
        lines = [_format_video_stats(info)]

        total_frames = info["total_frames"]
        h = info["height"]
        w = info["width"]
        bitrate = info.get("bitrate_kbps", 0)
        codec = info.get("codec", "")

        # Mirror the adaptive scale logic from upscale_hdr.py
        efficient_codecs = {"hevc", "h265", "av1", "vp9"}
        codec_bonus = 1.4 if codec.lower() in efficient_codecs else 1.0
        effective_bitrate = bitrate * codec_bonus
        pixels = h * w
        bpp = (effective_bitrate * 1000) / (pixels * 30) if pixels > 0 else 0

        if h <= 360:
            target_h = 1080 if bpp > 0.04 else 720
        elif h <= 480:
            target_h = 2160 if bpp > 0.08 else (1440 if bpp > 0.03 else 1080)
        elif h <= 720:
            target_h = 2160 if bpp > 0.06 else (1440 if bpp > 0.03 else 1080)
        elif h <= 1080:
            target_h = 2160
        elif h <= 1440:
            target_h = 2160
        else:
            target_h = h
        target_h = min(target_h, 2160)
        scale = max(target_h / h, 1.0)
        output_w = int(w * scale)
        output_h = target_h

        quality = "clean" if bpp > 0.06 else ("moderate" if bpp > 0.03 else "compressed")

        lines.append("")
        lines.append(f"**Source quality:** {quality} (bpp={bpp:.3f}, {codec})")
        lines.append(f"**Output resolution:** {output_w}x{output_h} ({scale:.1f}x upscale)")

        # Endpoint-aware cost/time estimates
        best, worst, gpu_list = _get_endpoint_gpu_specs(endpoint_choice)
        n_gpus = max(1, int(split_gpus))
        hdr_mult = 0.7 if no_itm else 1.0

        lines.append("")
        if gpu_list:
            gpu_names = [s[1]["short"] for s in gpu_list]
            lines.append(f"**GPU pool:** {', '.join(gpu_names)}")
        if n_gpus > 1:
            lines.append(f"**Workers:** {n_gpus} (split into {n_gpus} chunks of ~{total_frames // n_gpus:,} frames)")

        # Best case (fastest GPU)
        best_render = total_frames * best["spf"] * hdr_mult
        best_time = best_render / n_gpus + STARTUP_OVERHEAD_S
        best_cost = best_render * best["cost_s"] + (STARTUP_OVERHEAD_S * best["cost_s"] * n_gpus)
        # Worst case (slowest GPU)
        worst_render = total_frames * worst["spf"] * hdr_mult
        worst_time = worst_render / n_gpus + STARTUP_OVERHEAD_S
        worst_cost = worst_render * worst["cost_s"] + (STARTUP_OVERHEAD_S * worst["cost_s"] * n_gpus)

        lines.append("")
        if best["spf"] == worst["spf"]:
            lines.append(f"**Est. time:** ~{_fmt_duration(best_time)} ({best['short']} @ {best['spf']}s/frame)")
            lines.append(f"**Est. cost:** ~${best_cost:.2f}")
        else:
            lines.append(f"**Best case:** ~{_fmt_duration(best_time)} / ${best_cost:.2f} ({best['short']} @ {best['spf']}s/frame)")
            lines.append(f"**Worst case:** ~{_fmt_duration(worst_time)} / ${worst_cost:.2f} ({worst['short']} @ {worst['spf']}s/frame)")

        if no_itm:
            lines.append("*(SDR only — ~30% faster, no HDR stage)*")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def _fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"

def _recommend_settings(info):
    """Return recommended pipeline settings based on video analysis."""
    h = info.get("height", 0)
    w = info.get("width", 0)
    bitrate = info.get("bitrate_kbps", 0)
    codec = info.get("codec", "")
    duration = info.get("duration", 0)
    total_frames = info.get("total_frames", 0)

    # Denoise: auto for most, higher for low bitrate
    dn = -1  # auto

    # Tile: based on output resolution
    efficient_codecs = {"hevc", "h265", "av1", "vp9"}
    codec_bonus = 1.4 if codec.lower() in efficient_codecs else 1.0
    effective_bitrate = bitrate * codec_bonus
    pixels = h * w
    bpp = (effective_bitrate * 1000) / (pixels * 30) if pixels > 0 else 0

    # Target height (mirrors adaptive scale)
    if h <= 360:
        target_h = 1080 if bpp > 0.04 else 720
    elif h <= 480:
        target_h = 2160 if bpp > 0.08 else (1440 if bpp > 0.03 else 1080)
    elif h <= 720:
        target_h = 2160 if bpp > 0.06 else (1440 if bpp > 0.03 else 1080)
    elif h <= 1080:
        target_h = 2160
    elif h <= 1440:
        target_h = 2160
    else:
        target_h = h
    target_h = min(target_h, 2160)

    # Tile size: larger input = smaller tiles to avoid OOM
    output_h = target_h
    if output_h >= 2160:
        tile = 512
    else:
        tile = 1024

    # CRF: higher quality for clean sources
    crf = 16 if bpp > 0.06 else 18

    # Preset: always p7 — encoding is <1% of total pipeline time
    preset = "p7"

    # Workers: based on total processing work vs cold start overhead.
    # v34 benchmarks: ~1.55s/frame on 4090 at 4K output.
    # Each chunk has ~120s overhead (cold start + model load + upload).
    # Only split when total time saved > overhead cost.
    output_w = int(target_h * (w / h)) if h > 0 else target_h * 16 // 9
    output_mpx = (target_h * output_w) / 1e6
    est_sec_per_frame = 1.55 * (output_mpx / 8.3)  # ~1.55s at 4K (8.3 MP)
    est_total_sec = total_frames * est_sec_per_frame
    # Real overhead per chunk: cold start (~120s) + input download (~30s) +
    # output upload (~30s) + stitch time (~30s) + risk of bad GPU node
    cold_start_overhead = 300  # conservative — real-world measured

    time_1 = est_total_sec
    time_2 = est_total_sec / 2 + cold_start_overhead
    time_3 = est_total_sec / 3 + cold_start_overhead

    # Only split when it's clearly faster AND video is long enough.
    # 3-way needs >4500 frames (~2.5 min 60fps) — below that the overhead
    # of 3 cold starts eats the savings.  The time_3 < time_2 * 0.8 guard
    # already ensures the math works out before recommending 3.
    if time_2 < time_1 * 0.7 and total_frames > 3000:  # >3000 frames (~2 min 60fps)
        if time_3 < time_2 * 0.8 and total_frames > 4500:
            split = 3
        else:
            split = 2
    else:
        split = 1

    return {
        "dn": dn,
        "tile": tile,
        "crf": crf,
        "preset": preset,
        "target_height": target_h,
        "split": split,
    }


def probe_selected_files(files, endpoint_choice="", split_gpus=1, no_itm=False):
    """Probe files and return text + recommended settings for UI controls."""
    if not files:
        return "No files selected.", -1, 1024, 18, "p7", 0, 1
    results = []
    last_settings = {}
    for f in (files if isinstance(files, list) else [files]):
        path = f if isinstance(f, str) else f.name if hasattr(f, 'name') else str(f)
        results.append(f"### {os.path.basename(path)}")
        results.append(probe_video(path, endpoint_choice, split_gpus, no_itm))
        try:
            info = _probe_file(path)
            last_settings = _recommend_settings(info)
            rec = last_settings
            results.append(f"\n**Recommended:** tile={rec['tile']}, crf={rec['crf']}, "
                          f"preset={rec['preset']}, target={rec['target_height']}p, "
                          f"workers={rec['split']}")
        except:
            pass
        results.append("---")

    text = "\n".join(results)
    dn = last_settings.get("dn", -1)
    tile = last_settings.get("tile", 1024)
    crf = last_settings.get("crf", 18)
    preset = last_settings.get("preset", "p7")
    target = last_settings.get("target_height", 0)
    split = last_settings.get("split", 1)

    return text, dn, tile, crf, preset, target, split

# ─── API Helpers ─────────────────────────────────────────────────────────────

def get_health(endpoint_id):
    try:
        r = requests.get(f"{_api_base(endpoint_id)}/health",
                         headers=_headers(), timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def submit_job(endpoint_id, video_url, filename, params):
    safe_stem = re.sub(r'[|&;<>\"\'`$\\!#(){}[\]*?~]', '_',
                       os.path.splitext(filename)[0])
    suffix = "_sdr" if params.get("no_itm") else "_hdr"
    input_data = {
        "video_url": video_url,
        "output_filename": params.get("output_filename", f"{safe_stem}{suffix}.mkv"),
        "tile": params.get("tile", 1024),
        "batch": params.get("batch", 4),
        "workers": params.get("workers", 16),
    }
    # All optional params
    for key in ("dn", "face_strength", "itm", "hdr_mode", "crf", "preset",
                "target", "deinterlace"):
        if key in params and params[key] is not None:
            input_data[key] = params[key]
    for key in ("no_face", "no_itm", "fp16", "force_full_range", "preserve_grain", "no_trt"):
        if params.get(key):
            input_data[key] = True
    if "grain_strength" in params:
        input_data["grain_strength"] = float(params["grain_strength"])
    # Chunk params
    if "start_time" in params:
        input_data["start_time"] = params["start_time"]
    if "chunk_duration" in params:
        input_data["chunk_duration"] = params["chunk_duration"]

    timeout_ms = params.get("execution_timeout", 1800) * 1000
    policy = {"executionTimeout": timeout_ms, "ttl": 86400000}
    if params.get("low_priority"):
        policy["lowPriority"] = True
    payload = {
        "input": input_data,
        "policy": policy,
    }
    # Webhook: if configured, RunPod POSTs result on completion (retries 2x @ 10s)
    webhook_url = params.get("webhook_url", "")
    if webhook_url:
        payload["webhook"] = webhook_url
    r = requests.post(f"{_api_base(endpoint_id)}/run",
                      headers=_headers(), json=payload, timeout=30)
    return r.json()

def get_status(endpoint_id, job_id):
    try:
        r = requests.get(f"{_api_base(endpoint_id)}/status/{job_id}",
                         headers=_headers(), timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}

def cancel_job(endpoint_id, job_id):
    try:
        r = requests.post(f"{_api_base(endpoint_id)}/cancel/{job_id}",
                          headers=_headers(), timeout=10)
        return r.json()
    except Exception:
        return {"error": "Failed to cancel"}

def retry_job(endpoint_id, job_id):
    """Retry a failed job using the same job ID (preserves tracking)."""
    try:
        r = requests.post(f"{_api_base(endpoint_id)}/retry/{job_id}",
                          headers=_headers(), timeout=10)
        return r.json()
    except Exception:
        return {"error": "Failed to retry"}

def purge_queue(endpoint_id):
    try:
        r = requests.post(f"{_api_base(endpoint_id)}/purge-queue",
                          headers=_headers(), timeout=10)
        return r.json()
    except Exception:
        return {"error": "Failed to purge"}

# ─── Job Tracking ────────────────────────────────────────────────────────────

active_jobs = {}  # job_id -> {filename, status, endpoint_id, start_time, ...}
_jobs_lock = threading.Lock()

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"

def format_job_table():
    with _jobs_lock:
        if not active_jobs:
            return "No active jobs."
        rows = []
        for jid, info in sorted(active_jobs.items(),
                                 key=lambda x: x[1].get("start_time", 0),
                                 reverse=True):
            status = info.get("status", "UNKNOWN")
            filename = info.get("filename", "?")
            elapsed = time.time() - info.get("start_time", time.time())

            if status == "IN_PROGRESS":
                progress = info.get("progress", "")
                detail = f"{progress} | {format_time(elapsed)}"
                eta = info.get("eta", "")
                if eta:
                    detail += f" | ETA: {eta}"
                icon = "🔄"
            elif status == "IN_QUEUE":
                icon = "⏳"
                detail = f"Waiting ({format_time(elapsed)})"
            elif status == "COMPLETED":
                exec_time = info.get("execution_time", 0)
                icon = "✅"
                detail = f"Done in {format_time(exec_time)}"
            elif status == "FAILED":
                icon = "❌"
                detail = info.get("error", "Unknown error")[:80]
            elif status == "CANCELLED":
                icon = "🚫"
                detail = "Cancelled"
            else:
                icon = "❓"
                detail = status

            # Stats
            stats_lines = ""
            input_stats = info.get("input_stats", "")
            output_stats = info.get("output_stats", "")
            if input_stats:
                stats_lines += f"\n  📥 Input: {input_stats}"
            if output_stats:
                stats_lines += f"\n  📤 Output: {output_stats}"

            rows.append(f"{icon} **{filename}** — {detail}{stats_lines}  \n`{jid[:24]}...`\n")
        return "\n".join(rows)

def format_health_md(endpoint_choice):
    endpoint_id = _extract_endpoint_id(endpoint_choice)
    if not endpoint_id:
        return "No endpoint selected."
    h = get_health(endpoint_id)
    if "error" in h:
        return f"❌ {h['error']}"
    jobs = h.get("jobs", {})
    workers = h.get("workers", {})
    return f"""| Metric | Value |
|--------|-------|
| 🔄 In Progress | **{jobs.get('inProgress', 0)}** |
| ⏳ In Queue | **{jobs.get('inQueue', 0)}** |
| ✅ Completed | **{jobs.get('completed', 0)}** |
| ❌ Failed | **{jobs.get('failed', 0)}** |
| 🟢 Running | **{workers.get('running', 0)}** |
| 💤 Idle | **{workers.get('idle', 0)}** |
| 🔧 Initializing | **{workers.get('initializing', 0)}** |
| ⚠️ Throttled | **{workers.get('throttled', 0)}** |"""

# ─── Processing ──────────────────────────────────────────────────────────────

def process_videos(
    files, endpoint_choice,
    no_face, no_itm, no_denoise, tile, batch,
    dn, face_strength, highlight_boost, temporal_smooth, itm_model, hdr_mode,
    crf, enc_preset, target_height, deinterlace,
    fp16, force_full_range, preserve_grain, grain_strength, no_trt,
    workers, split_gpus, low_priority=False, webhook_url="",
    r2_selections=None,
    progress=gr.Progress(),
):
    endpoint_id = _extract_endpoint_id(endpoint_choice)
    if not endpoint_id:
        yield "❌ No endpoint selected.", format_health_md(endpoint_choice), format_job_table(), format_output_info()
        return

    # Combine local files + R2 selections
    file_list = []
    r2_urls = {}  # filename -> r2 url (skip upload for these)
    if files:
        file_list = files if isinstance(files, list) else [files]
    if r2_selections:
        for label in (r2_selections if isinstance(r2_selections, list) else [r2_selections]):
            if not label:
                continue
            fname = label.split(" (")[0]
            r2_urls[fname] = f"{BUCKET_ENDPOINT_URL}/{BUCKET_NAME}/input/{fname}"
            file_list.append(fname)

    if not file_list:
        yield "❌ No files selected.", format_health_md(endpoint_choice), format_job_table(), format_output_info()
        return
    log_lines = []

    def log(msg):
        log_lines.append(f"`{datetime.now().strftime('%H:%M:%S')}` {msg}")
        return "\n".join(log_lines[-40:])

    # ── Show job summary before processing ──
    num_gpus = int(split_gpus)
    summary_lines = [
        "## 📋 Job Summary",
        f"**Files:** {len(file_list)}",
    ]
    for f in file_list:
        fname = f if isinstance(f, str) else os.path.basename(f.name if hasattr(f, 'name') else str(f))
        src = "R2" if fname in r2_urls else "Local"
        summary_lines.append(f"  - `{fname}` ({src})")
    summary_lines.extend([
        "",
        "**Pipeline Settings:**",
        f"  - Denoise: {'auto' if float(dn) == -1 else ('skip' if float(dn) == 0 else dn)}",
        f"  - Face restore: {'OFF' if no_face else f'ON (strength={face_strength})'}",
        f"  - ITM/HDR: {'OFF (SDR output)' if no_itm else f'{itm_model} → {hdr_mode}'}",
        f"  - FP16: {'✅' if fp16 else '❌'}",
        f"  - Full range: {'forced' if force_full_range else 'auto'}",
        f"  - Preserve grain: {'ON' if preserve_grain else 'OFF'}",
        "",
        "**Encoding:**",
        f"  - Target: {target_height}p | CRF: {crf} | Preset: {enc_preset}",
        f"  - Tile: {tile} (auto on worker) | Batch: {batch}",
        "",
        f"**Workers:** {num_gpus} {'(split into chunks)' if num_gpus > 1 else '(single job)'}",
        f"**Endpoint:** `{endpoint_id}`",
        "",
        "---",
        "",
    ])
    output = "\n".join(summary_lines)
    yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()

    # Build params dict
    params = {
        "tile": int(tile),
        "batch": int(batch),
        "workers": int(workers),
        "no_face": no_face,
        "no_itm": no_itm,
        "no_denoise": no_denoise,
        "fp16": fp16,
        "force_full_range": force_full_range,
        "preserve_grain": preserve_grain,
        "no_trt": no_trt,
        "execution_timeout": 1800,
        "low_priority": low_priority,
        "webhook_url": webhook_url or "",
    }
    if dn >= 0:
        params["dn"] = dn
    else:
        params["dn"] = -1
    if face_strength >= 0:
        params["face_strength"] = face_strength
    if highlight_boost > 0:
        params["highlight_boost"] = float(highlight_boost)
    if temporal_smooth != 0.15:
        params["temporal_smooth"] = float(temporal_smooth)
    if preserve_grain and grain_strength > 0:
        params["grain_strength"] = float(grain_strength)
    if itm_model != "Auto":
        params["itm"] = itm_model
    if hdr_mode:
        params["hdr_mode"] = hdr_mode
    if crf > 0:
        params["crf"] = int(crf)
    if enc_preset:
        params["preset"] = enc_preset
    if int(target_height) > 0:
        params["target"] = int(target_height)
    if deinterlace != "auto":
        params["deinterlace"] = deinterlace

    num_gpus = int(split_gpus)
    all_job_ids = []

    for i, filepath in enumerate(file_list):
        path = filepath if isinstance(filepath, str) else filepath.name
        filename = os.path.basename(path)
        progress((i, len(file_list)), desc=f"Uploading {filename}...")

        # Probe input file stats
        input_info = None
        input_stats = ""
        try:
            if filename in r2_urls:
                # R2 file — download to temp for probing
                import tempfile
                tmp = Path(tempfile.mkdtemp()) / filename
                try:
                    subprocess.run(
                        ["rclone", "copyto", f"s3:{BUCKET_NAME}/input/{filename}", str(tmp)],
                        env=_r2_env(), capture_output=True, timeout=120)
                    if tmp.exists():
                        input_info = _probe_file(str(tmp))
                        input_stats = _format_video_stats(input_info)
                finally:
                    if tmp.exists():
                        tmp.unlink()
                    tmp.parent.rmdir()
            else:
                input_info = _probe_file(path)
                input_stats = _format_video_stats(input_info)
        except:
            pass

        # Check if file is from R2 (skip upload)
        if filename in r2_urls:
            video_url = r2_urls[filename]
            output = log(f"☁️ **{filename}** — already in R2, skipping upload")
            if input_stats:
                output = log(f"  📥 {input_stats}")
            yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()
        else:
            output = log(f"📤 Uploading **{filename}**")
            if input_stats:
                output = log(f"  📥 {input_stats}")
            yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()

            try:
                video_url = upload_to_r2(path)
                output = log(f"✅ Uploaded {filename}")
                yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()
            except Exception as e:
                output = log(f"❌ Upload failed: {e}")
                yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()
                continue

        # Smart split override: auto-correct based on actual video analysis
        effective_gpus = num_gpus
        if input_info and num_gpus > 1:
            total_frames = input_info.get("total_frames", 0)
            # Hard minimum: never split below 300 frames regardless of settings
            if total_frames < 300:
                effective_gpus = 1
                output = log(f"⚡ Auto-adjusted split: {num_gpus} → 1 "
                             f"({total_frames} frames — too short to split)")
                yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()
            else:
                rec = _recommend_settings(input_info)
                recommended_split = rec.get("split", 1)
                if recommended_split < num_gpus:
                    effective_gpus = recommended_split
                    output = log(f"⚡ Auto-adjusted split: {num_gpus} → {effective_gpus} "
                                 f"({total_frames} frames, "
                                 f"overhead not worth splitting further)")
                    yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()

        if effective_gpus > 1:
            # Multi-GPU split
            try:
                if not input_info:
                    output = log(f"❌ Cannot split — failed to probe {filename}")
                    yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()
                    continue
                duration = input_info["duration"]
                chunk_dur = duration / effective_gpus

                for g in range(effective_gpus):
                    start = g * chunk_dur
                    dur = chunk_dur if g < effective_gpus - 1 else duration - start
                    chunk_params = {
                        **params,
                        "start_time": round(start, 3),
                        "chunk_duration": round(dur, 3),
                    }
                    safe_stem = _sanitize_s3_key(os.path.splitext(filename)[0])
                    chunk_params["output_filename"] = f"{safe_stem}_chunk_{g+1}.mkv"

                    result = submit_job(endpoint_id, video_url, filename, chunk_params)
                    job_id = result.get("id", "")
                    if job_id:
                        all_job_ids.append(job_id)
                        chunk_frames = int(dur * input_info.get("fps", 30))
                        with _jobs_lock:
                            active_jobs[job_id] = {
                                "filename": f"{filename} (chunk {g+1}/{effective_gpus})",
                                "status": "IN_QUEUE",
                                "endpoint_id": endpoint_id,
                                "start_time": time.time(),
                                "is_chunk": True,
                                "chunk_index": g,
                                "total_chunks": effective_gpus,
                                "parent_file": filename,
                                "input_stats": input_stats,
                                "chunk_frames": chunk_frames,
                            }
                        output = log(f"🚀 Chunk {g+1}/{effective_gpus} → `{job_id[:20]}...` ({chunk_frames:,} frames)")
                    else:
                        output = log(f"❌ Chunk {g+1} submit failed: {result}")
                    yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()
            except Exception as e:
                output = log(f"❌ Split failed: {e}")
                yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()
                continue
        else:
            # Single job
            try:
                result = submit_job(endpoint_id, video_url, filename, params)
                job_id = result.get("id", "")
                if job_id:
                    all_job_ids.append(job_id)
                    with _jobs_lock:
                        active_jobs[job_id] = {
                            "filename": filename,
                            "status": "IN_QUEUE",
                            "endpoint_id": endpoint_id,
                            "start_time": time.time(),
                            "input_stats": input_stats,
                        }
                    output = log(f"🚀 Submitted **{filename}** → `{job_id[:20]}...`")
                else:
                    output = log(f"❌ Submit failed: {result}")
                yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()
            except Exception as e:
                output = log(f"❌ Submit error: {e}")
                yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()

    if not all_job_ids:
        output = log("❌ No jobs submitted.")
        yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()
        return

    # Show cost estimate
    best, worst, gpu_list = _get_endpoint_gpu_specs(endpoint_choice)
    output = log(f"\n📊 **{len(all_job_ids)} job(s) submitted** — polling...")
    yield output, format_health_md(endpoint_choice), format_job_table(), format_output_info()

    # Poll all jobs with adaptive interval
    completed = set()
    failed = set()
    poll_start = time.time()

    while len(completed) + len(failed) < len(all_job_ids):
        # Adaptive polling: fast at first, slower for long jobs
        elapsed = time.time() - poll_start
        if elapsed < 30:
            poll_interval = 3
        elif elapsed < 120:
            poll_interval = 5
        elif elapsed < 600:
            poll_interval = 10
        else:
            poll_interval = 20
        time.sleep(poll_interval)
        for jid in all_job_ids:
            if jid in completed or jid in failed:
                continue
            status_data = get_status(endpoint_id, jid)
            status = status_data.get("status", "UNKNOWN")
            with _jobs_lock:
                info = active_jobs.get(jid, {})
                info["status"] = status

                if status == "IN_PROGRESS":
                    out = status_data.get("output", "")
                    if isinstance(out, str):
                        info["progress"] = out
                        if "%" in out:
                            try:
                                pct = float(out.split("%")[0].split()[-1])
                                elapsed = time.time() - info.get("start_time", time.time())
                                if pct > 0:
                                    remaining = (elapsed / pct) * (100 - pct)
                                    info["eta"] = format_time(remaining)
                            except (ValueError, IndexError):
                                pass

                elif status == "COMPLETED":
                    completed.add(jid)
                    exec_time = status_data.get("executionTime", 0) / 1000
                    info["execution_time"] = exec_time
                    output_data = status_data.get("output", {})

                    out_filename = "output.mkv"
                    if isinstance(output_data, dict):
                        out_filename = output_data.get("filename", "output.mkv")

                    # Try presigned URL first, fallback to rclone from R2
                    out_path = OUTPUT_DIR / out_filename
                    downloaded = False

                    if isinstance(output_data, dict) and output_data.get("output_url"):
                        try:
                            r = requests.get(output_data["output_url"], stream=True, timeout=600)
                            r.raise_for_status()
                            with open(out_path, "wb") as f:
                                for chunk in r.iter_content(8192):
                                    f.write(chunk)
                            if out_path.exists() and out_path.stat().st_size > 0:
                                downloaded = True
                        except Exception:
                            pass

                    # Fallback: download from R2 via rclone
                    if not downloaded:
                        try:
                            s3_key = f"output/{jid}/{out_filename}"
                            subprocess.run(
                                ["rclone", "copyto", f"s3:{BUCKET_NAME}/{s3_key}", str(out_path)],
                                env=_r2_env(), capture_output=True, timeout=600)
                            if out_path.exists() and out_path.stat().st_size > 0:
                                downloaded = True
                        except Exception:
                            pass

                    if downloaded:
                        try:
                            out_info = _probe_file(out_path)
                            info["output_stats"] = _compact_stats(out_info)
                            info["output_stats_full"] = _format_video_stats(out_info)
                        except:
                            size_mb = out_path.stat().st_size / 1048576
                            info["output_stats"] = f"{size_mb:.1f} MB"
                        output = log(f"✅ **{out_filename}** ({format_time(exec_time)})")
                        if info.get("output_stats"):
                            output = log(f"  📤 {info['output_stats']}")
                    else:
                        output = log(f"⚠️ Auto-download failed for {out_filename} — use Download Results button")

                elif status == "FAILED":
                    failed.add(jid)
                    error = status_data.get("error", "Unknown error")
                    info["error"] = error
                    output = log(f"❌ **{info.get('filename')}** failed: {error[:100]}")

                elif status == "CANCELLED":
                    failed.add(jid)
                    info["status"] = "CANCELLED"
                    output = log(f"🚫 **{info.get('filename')}** cancelled")

                active_jobs[jid] = info

        yield "\n".join(log_lines[-40:]), format_health_md(endpoint_choice), format_job_table(), format_output_info()

    # Auto-stitch any completed chunk sets
    stitched = _auto_stitch_chunks(OUTPUT_DIR)
    if stitched:
        for s in stitched:
            output = log(f"🔗 {s}")
        # Update output stats for stitched file
        yield "\n".join(log_lines[-40:]), format_health_md(endpoint_choice), format_job_table(), format_output_info()

    # Summary
    output = log(f"\n{'='*50}")
    output = log(f"📊 **Done:** {len(completed)} succeeded, {len(failed)} failed")
    if stitched:
        output = log(f"🔗 Stitched {len(stitched)} video(s)")
    output = log(f"📁 Results in `{OUTPUT_DIR.absolute()}`")
    yield "\n".join(log_lines[-40:]), format_health_md(endpoint_choice), format_job_table(), format_output_info()

# ─── Management Functions ────────────────────────────────────────────────────

def do_refresh_health(endpoint_choice):
    return format_health_md(endpoint_choice)

def do_refresh_jobs(endpoint_choice):
    endpoint_id = _extract_endpoint_id(endpoint_choice)
    with _jobs_lock:
        for jid, info in list(active_jobs.items()):
            if info.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
                continue
            eid = info.get("endpoint_id", endpoint_id)
            status_data = get_status(eid, jid)
            info["status"] = status_data.get("status", "UNKNOWN")
            out = status_data.get("output", "")
            if isinstance(out, str) and out:
                info["progress"] = out
    return format_job_table()

def format_output_info():
    """Show full stats for all completed jobs, most recent first."""
    with _jobs_lock:
        completed = [(jid, info) for jid, info in active_jobs.items()
                     if info.get("status") == "COMPLETED"]
    if not completed:
        return "No completed jobs yet."
    completed.sort(key=lambda x: x[1].get("start_time", 0), reverse=True)
    sections = []
    for jid, info in completed:
        filename = info.get("filename", "?")
        exec_time = info.get("execution_time", 0)
        parts = [f"#### {filename}", f"**Render time:** {format_time(exec_time)}"]
        input_stats = info.get("input_stats", "")
        if input_stats:
            parts.append(f"\n**📥 Input**\n{input_stats}")
        output_full = info.get("output_stats_full", "")
        if output_full:
            parts.append(f"\n**📤 Output**\n{output_full}")
        elif info.get("output_stats"):
            parts.append(f"\n**📤 Output:** {info['output_stats']}")
        sections.append("\n".join(parts))
    return "\n\n---\n\n".join(sections)


def do_cancel_job(endpoint_choice, job_id_text):
    endpoint_id = _extract_endpoint_id(endpoint_choice)
    if not job_id_text.strip():
        return "Enter a job ID."
    result = cancel_job(endpoint_id, job_id_text.strip())
    with _jobs_lock:
        if job_id_text.strip() in active_jobs:
            active_jobs[job_id_text.strip()]["status"] = "CANCELLED"
    return json.dumps(result, indent=2)

def do_cancel_all(endpoint_choice):
    endpoint_id = _extract_endpoint_id(endpoint_choice)
    cancelled = 0
    with _jobs_lock:
        for jid, info in active_jobs.items():
            if info.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
                cancel_job(info.get("endpoint_id", endpoint_id), jid)
                info["status"] = "CANCELLED"
                cancelled += 1
    purge_queue(endpoint_id)
    return f"Cancelled {cancelled} job(s) and purged queue."

def do_purge(endpoint_choice):
    endpoint_id = _extract_endpoint_id(endpoint_choice)
    result = purge_queue(endpoint_id)
    removed = result.get("removed", 0)
    return f"Purged {removed} job(s) from queue."

def do_retry_failed(endpoint_choice):
    """Retry all failed jobs using /retry endpoint (preserves job IDs)."""
    endpoint_id = _extract_endpoint_id(endpoint_choice)
    retried = 0
    errors = 0
    with _jobs_lock:
        for jid, info in active_jobs.items():
            if info.get("status") == "FAILED":
                result = retry_job(info.get("endpoint_id", endpoint_id), jid)
                if "error" not in result:
                    info["status"] = "IN_QUEUE"
                    info["start_time"] = time.time()
                    retried += 1
                else:
                    errors += 1
    msg = f"Retried {retried} failed job(s)."
    if errors:
        msg += f" {errors} could not be retried (may have expired)."
    return msg

def do_download_results(endpoint_choice):
    """Download all completed outputs from R2."""
    try:
        result = subprocess.run(
            ["rclone", "ls", f"s3:{BUCKET_NAME}/output/"],
            env=_r2_env(), capture_output=True, text=True, timeout=30)
        if not result.stdout.strip():
            return "No outputs found in R2."

        lines = result.stdout.strip().split("\n")
        downloaded = []
        for line in lines:
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue
            s3_path = parts[1]
            filename = os.path.basename(s3_path)
            local_path = OUTPUT_DIR / filename
            # Don't skip existing — R2 overwrites, but increment local non-chunk files
            if local_path.exists() and "_chunk_" not in filename:
                stem, ext_ = os.path.splitext(filename)
                counter = 2
                while local_path.exists():
                    local_path = OUTPUT_DIR / f"{stem}_{counter}{ext_}"
                    counter += 1
            subprocess.run(
                ["rclone", "copyto", f"s3:{BUCKET_NAME}/output/{s3_path}", str(local_path)],
                env=_r2_env(), capture_output=True, timeout=600)
            if local_path.exists():
                try:
                    stats = _compact_stats(_probe_file(local_path))
                    downloaded.append(f"✅ {filename}\n  ↳ {stats}")
                except:
                    size_mb = local_path.stat().st_size / 1048576
                    downloaded.append(f"✅ {filename} ({size_mb:.1f} MB)")

        # Auto-stitch chunks
        stitched = _auto_stitch_chunks(OUTPUT_DIR)

        msg_parts = []
        if downloaded:
            msg_parts.append(f"Downloaded {len(downloaded)} file(s):\n" + "\n".join(downloaded))
        if stitched:
            msg_parts.append(f"\nStitched {len(stitched)} video(s):\n" + "\n".join(stitched))
        if msg_parts:
            return "\n".join(msg_parts)
        else:
            return "All outputs already downloaded."
    except Exception as e:
        return f"Error: {e}"


def _auto_stitch_chunks(output_dir):
    """Find chunk sets in output_dir and stitch them with ffmpeg concat.
    Only stitches when all expected chunks are present (checks job tracker)."""
    import re as _re
    chunk_pattern = _re.compile(r'^(.+)_chunk_(\d+)\.(mkv|mp4)$')
    groups = {}
    for f in sorted(output_dir.iterdir()):
        m = chunk_pattern.match(f.name)
        if m:
            base, idx, ext = m.group(1), int(m.group(2)), m.group(3)
            key = (base, ext)
            if key not in groups:
                groups[key] = []
            groups[key].append((idx, f))

    # Build expected chunk counts from job tracker
    expected_chunks = {}  # base_name -> total_chunks
    with _jobs_lock:
        for jid, info in active_jobs.items():
            if info.get("is_chunk") and info.get("total_chunks"):
                parent = info.get("parent_file", "")
                safe_stem = _sanitize_s3_key(os.path.splitext(parent)[0])
                expected_chunks[safe_stem] = info["total_chunks"]

    stitched = []
    for (base, ext), chunks in groups.items():
        final_name = f"{base}_hdr.{ext}"
        final_path = output_dir / final_name
        # Auto-increment if exists locally
        counter = 2
        while final_path.exists():
            final_name = f"{base}_hdr_{counter}.{ext}"
            final_path = output_dir / final_name
            counter += 1
        chunks.sort(key=lambda x: x[0])
        indices = [c[0] for c in chunks]

        # Check we have all chunks — use job tracker if available, else require no gaps
        total_expected = expected_chunks.get(base, max(indices))
        if indices != list(range(1, total_expected + 1)):
            continue
        # Write concat list
        concat_file = output_dir / f".concat_{base}.txt"
        with open(concat_file, "w") as cf:
            for _, chunk_path in chunks:
                cf.write(f"file '{chunk_path.name}'\n")
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_file),
                 "-c", "copy", str(final_path), "-y"],
                capture_output=True, text=True, timeout=120, cwd=str(output_dir))
            if final_path.exists():
                size_mb = final_path.stat().st_size / 1048576
                try:
                    stats = _compact_stats(_probe_file(final_path))
                    stitched.append(f"🔗 {final_name}\n  ↳ {stats}")
                except:
                    stitched.append(f"🔗 {final_name} ({size_mb:.1f} MB)")
                # Upload stitched file to R2 under output/stitched/
                try:
                    r2_key = f"output/stitched/{final_name}"
                    subprocess.run(
                        ["rclone", "copyto", str(final_path), f"s3:{BUCKET_NAME}/{r2_key}"],
                        env=_r2_env(), capture_output=True, timeout=600)
                    stitched[-1] += f"\n  ↳ ☁️ Uploaded to R2: {r2_key}"
                except Exception as e:
                    stitched[-1] += f"\n  ↳ ⚠️ R2 upload failed: {e}"
                # Clean up chunk files locally
                for _, chunk_path in chunks:
                    chunk_path.unlink(missing_ok=True)
        except Exception as e:
            stitched.append(f"❌ Failed to stitch {base}: {e}")
        finally:
            concat_file.unlink(missing_ok=True)

    return stitched


def do_browse_r2():
    """List all files in R2 bucket with sizes, grouped by folder, and flag active renders."""
    try:
        result = subprocess.run(
            ["rclone", "ls", f"s3:{BUCKET_NAME}/"],
            env=_r2_env(), capture_output=True, text=True, timeout=30)
        if not result.stdout.strip():
            return "R2 bucket is empty."

        # Collect currently rendering filenames from active jobs
        rendering = set()
        with _jobs_lock:
            for jid, info in active_jobs.items():
                if info.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
                    rendering.add(info.get("filename", ""))
                    # Also add sanitized name for chunk matches
                    rendering.add(_sanitize_s3_key(info.get("filename", "")))

        # Parse rclone ls output: "  size path/to/file"
        folders = {}
        total_size = 0
        file_count = 0
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue
            size_bytes = int(parts[0])
            s3_path = parts[1]
            total_size += size_bytes
            file_count += 1

            folder = s3_path.split("/")[0] if "/" in s3_path else "(root)"
            if folder not in folders:
                folders[folder] = []

            filename = os.path.basename(s3_path)
            size_mb = size_bytes / 1048576

            # Check if this file is currently being rendered
            is_rendering = False
            for rname in rendering:
                if rname and (rname in filename or _sanitize_s3_key(rname) in filename):
                    is_rendering = True
                    break

            status = " 🔄 **RENDERING**" if is_rendering else ""
            folders[folder].append(f"  `{filename}` ({size_mb:.1f} MB){status}")

        lines = [f"**R2 Bucket:** `{BUCKET_NAME}` — {file_count} files, {total_size / 1048576:.0f} MB total\n"]
        for folder in sorted(folders.keys()):
            lines.append(f"📁 **/{folder}/** ({len(folders[folder])} files)")
            lines.extend(sorted(folders[folder]))
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def _list_r2_files():
    """Return list of (display_label, s3_path) for all R2 files."""
    result = subprocess.run(
        ["rclone", "ls", f"s3:{BUCKET_NAME}/"],
        env=_r2_env(), capture_output=True, text=True, timeout=30)
    files = []
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                size_bytes = int(parts[0])
                s3_path = parts[1]
                size_mb = size_bytes / 1048576
                label = f"{s3_path} ({size_mb:.1f} MB)"
                files.append(label)
    return files


def do_browse_r2_interactive():
    """Refresh R2 file list for the dropdown."""
    try:
        files = _list_r2_files()
        if not files:
            return gr.CheckboxGroup(choices=[], value=[]), "R2 bucket is empty."
        total_mb = sum(float(f.split("(")[-1].split(" MB")[0]) for f in files)
        return gr.CheckboxGroup(choices=files, value=[]), f"**{len(files)} files** — {total_mb:.0f} MB total"
    except Exception as e:
        return gr.CheckboxGroup(choices=[], value=[]), f"Error: {e}"


def do_r2_download_selected(selected_files):
    """Download selected files from R2."""
    if not selected_files:
        return "Select files first."
    downloaded = []
    for label in selected_files:
        s3_path = label.split(" (")[0]
        filename = os.path.basename(s3_path)
        local_path = OUTPUT_DIR / filename
        try:
            subprocess.run(
                ["rclone", "copyto", f"s3:{BUCKET_NAME}/{s3_path}", str(local_path)],
                env=_r2_env(), capture_output=True, timeout=600)
            if local_path.exists():
                try:
                    stats = _compact_stats(_probe_file(local_path))
                    downloaded.append(f"✅ {filename}\n  ↳ {stats}")
                except:
                    size_mb = local_path.stat().st_size / 1048576
                    downloaded.append(f"✅ {filename} ({size_mb:.1f} MB)")
        except Exception as e:
            downloaded.append(f"❌ {filename}: {e}")
    # Auto-stitch if chunks
    stitched = _auto_stitch_chunks(OUTPUT_DIR)
    if stitched:
        downloaded.extend(stitched)
    return "\n".join(downloaded) if downloaded else "No files downloaded."


def do_r2_delete_selected(selected_files):
    """Delete selected files from R2."""
    if not selected_files:
        return "Select files first."
    deleted = []
    for label in selected_files:
        s3_path = label.split(" (")[0]
        try:
            subprocess.run(
                ["rclone", "deletefile", f"s3:{BUCKET_NAME}/{s3_path}"],
                env=_r2_env(), capture_output=True, timeout=30)
            deleted.append(f"🗑 {s3_path}")
        except Exception as e:
            deleted.append(f"❌ {s3_path}: {e}")
    return "\n".join(deleted)


def do_r2_delete_all():
    """Delete everything in R2."""
    try:
        subprocess.run(
            ["rclone", "delete", f"s3:{BUCKET_NAME}/", "--rmdirs"],
            env=_r2_env(), capture_output=True, timeout=60)
        return "🗑 All R2 files deleted."
    except Exception as e:
        return f"Error: {e}"


def do_browse_folder(folder_path):
    """List video files in a folder with stats, plus R2 contents."""
    sections = []

    # Local folder
    if folder_path and os.path.isdir(folder_path):
        extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".m2ts", ".flv"}
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for f in sorted(filenames):
                if os.path.splitext(f)[1].lower() in extensions:
                    full = os.path.join(root, f)
                    size_mb = os.path.getsize(full) / 1048576
                    files.append(f"  `{f}` ({size_mb:.0f} MB)")
        if files:
            sections.append(f"**📁 Local — {len(files)} videos:**\n" + "\n".join(files[:50]))
        else:
            sections.append("**📁 Local:** No video files found.")

    # R2 bucket
    try:
        result = subprocess.run(
            ["rclone", "ls", f"s3:{BUCKET_NAME}/"],
            env=_r2_env(), capture_output=True, text=True, timeout=30)
        if result.stdout.strip():
            r2_files = {"input": [], "output": []}
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    size_bytes = int(parts[0])
                    s3_path = parts[1]
                    size_mb = size_bytes / 1048576
                    folder = s3_path.split("/")[0] if "/" in s3_path else "other"
                    filename = os.path.basename(s3_path)
                    bucket = folder if folder in r2_files else "output"
                    r2_files.setdefault(bucket, []).append(f"  `{filename}` ({size_mb:.1f} MB)")
            r2_lines = [f"**☁️ R2 Bucket** (`{BUCKET_NAME}`):"]
            for folder in sorted(r2_files.keys()):
                if r2_files[folder]:
                    r2_lines.append(f"\n**/{folder}/** ({len(r2_files[folder])} files)")
                    r2_lines.extend(sorted(r2_files[folder]))
            sections.append("\n".join(r2_lines))
        else:
            sections.append("**☁️ R2:** Empty")
    except Exception as e:
        sections.append(f"**☁️ R2:** Error — {e}")

    return "\n\n---\n\n".join(sections) if sections else "Enter a folder path."

def _list_r2_inputs():
    """List input files in R2 as dropdown choices."""
    try:
        result = subprocess.run(
            ["rclone", "ls", f"s3:{BUCKET_NAME}/input/"],
            env=_r2_env(), capture_output=True, text=True, timeout=30)
        choices = []
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    size_mb = int(parts[0]) / 1048576
                    filename = parts[1]
                    choices.append(f"{filename} ({size_mb:.1f} MB)")
        return choices
    except:
        return []


def do_refresh_r2_inputs():
    choices = _list_r2_inputs()
    if not choices:
        return gr.CheckboxGroup(choices=[], value=[]), "No input files in R2."
    return gr.CheckboxGroup(choices=choices, value=[]), f"**{len(choices)} input files** in R2"


def do_probe_r2_inputs(selected, endpoint_choice, split_gpus, no_itm):
    """Probe R2 input files — download to temp, probe, recommend settings, clean up."""
    if not selected:
        return "Select files first.", -1, 1024, 18, "p7", 0, 1
    results = []
    last_settings = {}
    for label in selected:
        filename = label.split(" (")[0]
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / filename
        try:
            s3_key = f"input/{filename}"
            subprocess.run(
                ["rclone", "copyto", f"s3:{BUCKET_NAME}/{s3_key}", str(tmp)],
                env=_r2_env(), capture_output=True, timeout=120)
            if tmp.exists():
                results.append(f"**{filename}**")
                results.append(probe_video(str(tmp), endpoint_choice, split_gpus, no_itm))
                try:
                    info = _probe_file(str(tmp))
                    last_settings = _recommend_settings(info)
                    rec = last_settings
                    results.append(f"\n**Recommended:** tile={rec['tile']}, crf={rec['crf']}, "
                                  f"preset={rec['preset']}, target={rec['target_height']}p, "
                                  f"workers={rec['split']}")
                except:
                    pass
                results.append("")
            else:
                results.append(f"**{filename}** — download failed")
        except Exception as e:
            results.append(f"**{filename}** — {e}")
        finally:
            if tmp.exists():
                tmp.unlink()
            try:
                tmp.parent.rmdir()
            except:
                pass

    text = "\n\n---\n\n".join(results) if results else "No files probed."
    dn = last_settings.get("dn", -1)
    tile = last_settings.get("tile", 1024)
    crf = last_settings.get("crf", 18)
    preset = last_settings.get("preset", "p7")
    target = last_settings.get("target_height", 0)
    split = last_settings.get("split", 1)
    return text, dn, tile, crf, preset, target, split


def do_requeue_r2_inputs(selected, endpoint_choice,
                          no_face, no_itm, no_denoise, tile, batch,
                          dn, face_strength, highlight_boost, temporal_smooth, itm_model, hdr_mode,
                          crf, enc_preset, target_height, deinterlace,
                          fp16, force_full_range, preserve_grain, grain_strength, no_trt,
                          dec_workers, split_gpus,
                          low_priority=False, webhook_url=""):
    """Submit jobs for files already in R2 (skip upload)."""
    if not selected:
        return "Select files first."
    endpoint_id = _extract_endpoint_id(endpoint_choice)
    if not endpoint_id:
        return "No endpoint selected."

    params = {
        "tile": int(tile), "batch": int(batch), "workers": int(dec_workers),
        "no_face": no_face, "no_itm": no_itm, "no_denoise": no_denoise, "fp16": fp16,
        "force_full_range": force_full_range,
        "preserve_grain": preserve_grain,
        "no_trt": no_trt,
        "execution_timeout": 1800,
        "low_priority": low_priority, "webhook_url": webhook_url or "",
    }
    if dn >= 0:
        params["dn"] = dn
    if face_strength != 0.5:
        params["face_strength"] = face_strength
    if highlight_boost > 0:
        params["highlight_boost"] = float(highlight_boost)
    if temporal_smooth != 0.15:
        params["temporal_smooth"] = float(temporal_smooth)
    if preserve_grain and grain_strength > 0:
        params["grain_strength"] = float(grain_strength)
    if itm_model != "Auto":
        params["itm"] = itm_model
    if hdr_mode != "hdr10":
        params["hdr_mode"] = hdr_mode
    if crf != 18:
        params["crf"] = int(crf)
    if enc_preset:
        params["preset"] = enc_preset
    if target_height != 0:
        params["target"] = int(target_height)
    if deinterlace and deinterlace != "auto":
        params["deinterlace"] = deinterlace

    num_gpus = int(split_gpus)
    results = []
    for label in selected:
        filename = label.split(" (")[0]  # strip size suffix
        video_url = f"{BUCKET_ENDPOINT_URL}/{BUCKET_NAME}/input/{filename}"

        # Probe for smart split override
        input_info = None
        try:
            import tempfile
            tmp = Path(tempfile.mkdtemp()) / filename
            subprocess.run(
                ["rclone", "copyto", f"s3:{BUCKET_NAME}/input/{filename}", str(tmp)],
                env=_r2_env(), capture_output=True, timeout=120)
            if tmp.exists():
                input_info = _probe_file(str(tmp))
                tmp.unlink()
            try:
                tmp.parent.rmdir()
            except:
                pass
        except:
            pass

        # Smart split override
        effective_gpus = num_gpus
        if input_info and num_gpus > 1:
            total_frames = input_info.get("total_frames", 0)
            if total_frames < 300:
                effective_gpus = 1
                results.append(f"⚡ {filename}: split {num_gpus} → 1 ({total_frames} frames — too short)")
            else:
                rec = _recommend_settings(input_info)
                recommended_split = rec.get("split", 1)
                if recommended_split < num_gpus:
                    effective_gpus = recommended_split
                    results.append(f"⚡ {filename}: split {num_gpus} → {effective_gpus}")

        if effective_gpus > 1 and input_info:
            # Multi-GPU split
            duration = input_info["duration"]
            chunk_dur = duration / effective_gpus
            for g in range(effective_gpus):
                start = g * chunk_dur
                dur = chunk_dur if g < effective_gpus - 1 else duration - start
                chunk_params = {
                    **params,
                    "start_time": round(start, 3),
                    "chunk_duration": round(dur, 3),
                }
                safe_stem = _sanitize_s3_key(os.path.splitext(filename)[0])
                chunk_params["output_filename"] = f"{safe_stem}_chunk_{g+1}.mkv"
                result = submit_job(endpoint_id, video_url, filename, chunk_params)
                job_id = result.get("id", "")
                if job_id:
                    chunk_frames = int(dur * input_info.get("fps", 30))
                    with _jobs_lock:
                        active_jobs[job_id] = {
                            "filename": f"{filename} (chunk {g+1}/{effective_gpus})",
                            "status": "IN_QUEUE",
                            "endpoint_id": endpoint_id,
                            "start_time": time.time(),
                            "is_chunk": True,
                            "chunk_index": g,
                            "total_chunks": effective_gpus,
                            "parent_file": filename,
                            "chunk_frames": chunk_frames,
                        }
                    results.append(f"🚀 {filename} chunk {g+1}/{effective_gpus} → `{job_id[:20]}...`")
                else:
                    results.append(f"❌ {filename} chunk {g+1} — {result}")
        else:
            result = submit_job(endpoint_id, video_url, filename, params)
            job_id = result.get("id", "")
            if job_id:
                with _jobs_lock:
                    active_jobs[job_id] = {
                        "filename": filename,
                        "status": "IN_QUEUE",
                        "endpoint_id": endpoint_id,
                        "start_time": time.time(),
                    }
                results.append(f"🚀 **{filename}** → `{job_id[:20]}...`")
            else:
                results.append(f"❌ **{filename}** — {result}")

    return "\n".join(results)


# ─── UI ──────────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Upscale & HDR Pipeline") as app:
    gr.Markdown("# 🎬 AI Upscale & HDR Pipeline\nLocal UI → RunPod Serverless")

    with gr.Row():
        # ════════════ LEFT COLUMN ════════════
        with gr.Column(scale=3):

            # ── Endpoint ──
            endpoint_choice = gr.Dropdown(
                label="RunPod Endpoint",
                choices=ENDPOINT_CHOICES,
                value=DEFAULT_ENDPOINT,
            )

            # ── Endpoint Info (auto-updates when endpoint changes) ──
            with gr.Row():
                endpoint_info = gr.Markdown(
                    value=_get_endpoint_info_md(DEFAULT_ENDPOINT) if DEFAULT_ENDPOINT else "No endpoints found.",
                    label="Endpoint Config",
                )
            refresh_gpu_btn = gr.Button("Refresh GPU Availability", size="sm")

            # ── Input ──
            with gr.Tab("File Upload"):
                files_input = gr.File(
                    label="Select video files",
                    file_count="multiple",
                    file_types=[".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm"],
                )
            with gr.Tab("Browse Folder"):
                with gr.Row():
                    folder_input = gr.Textbox(
                        label="Folder path",
                        placeholder="~/videos",
                        scale=3,
                    )
                    browse_btn = gr.Button("Browse", scale=1)
                folder_contents = gr.Markdown(value="")
                browse_btn.click(fn=do_browse_folder, inputs=[folder_input],
                                outputs=[folder_contents])
            with gr.Tab("R2 Input"):
                r2_input_select = gr.CheckboxGroup(
                    label="Select from R2 (already uploaded)",
                    choices=[], interactive=True,
                )
                with gr.Row():
                    r2_input_refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    r2_input_probe_btn = gr.Button("📊 Probe Selected", size="sm")
                    r2_input_requeue_btn = gr.Button("🚀 Requeue Selected", variant="primary", size="sm")
                r2_input_status = gr.Markdown(value="Click Refresh to list R2 inputs.")

            # ── Pipeline Options ──
            gr.Markdown("### Pipeline Options")
            with gr.Row():
                no_face = gr.Checkbox(label="Skip face restore", value=False)
                no_itm = gr.Checkbox(label="Skip HDR (SDR output)", value=False)
                no_denoise = gr.Checkbox(label="Skip denoise", value=False,
                                         info="Skip Stage 1 artifact cleanup (preserves original grain/noise)")
                fp16 = gr.Checkbox(label="FP16", value=True)
                force_full_range = gr.Checkbox(label="Force full range", value=False)
                preserve_grain = gr.Checkbox(label="Preserve grain", value=False,
                                             info="Re-apply film grain after processing (music videos, stylistic content)")
                no_trt = gr.Checkbox(label="Disable TensorRT", value=True, visible=False)  # always True on serverless

            with gr.Row():
                dn = gr.Slider(label="Denoise strength", minimum=-1, maximum=1,
                               step=0.05, value=-1,
                               info="-1 = auto from bitrate")
                face_strength = gr.Slider(label="Face strength", minimum=0, maximum=1,
                                          step=0.05, value=0.5,
                                          info="0 = aggressive, 1 = bypass")
                highlight_boost = gr.Slider(label="Highlight boost", minimum=0, maximum=1,
                                            step=0.05, value=0.0,
                                            info="Specular highlight expansion in ITM stage")
                temporal_smooth = gr.Slider(label="Temporal smooth", minimum=0, maximum=1,
                                            step=0.05, value=0.15,
                                            info="Reduce scene-transition brightness shifts (0=off)")
                grain_strength = gr.Slider(label="Grain strength", minimum=0, maximum=2,
                                           step=0.1, value=1.0, visible=False,
                                           info="1.0 = match original, >1.0 = amplify")

            with gr.Row():
                itm_model = gr.Dropdown(
                    label="ITM checkpoint (current: params_3DM)",
                    choices=["Auto", "params_3DM", "params_3DLUT", "params_DaVinci"],
                    value="Auto",
                    info="Auto = params_3DM (3D LUT + modulation, best quality)",
                )
                hdr_mode = gr.Radio(
                    label="HDR mode",
                    choices=["hdr10", "hdr10plus"],
                    value="hdr10",
                )

            with gr.Accordion("Encoding & Performance", open=False):
                with gr.Row():
                    tile = gr.Number(label="Tile size", value=1024, info="0 = full frame")
                    batch = gr.Number(label="ITM batch", value=4)
                    dec_workers = gr.Number(label="Decode workers", value=16)
                with gr.Row():
                    crf = gr.Slider(label="CRF", minimum=0, maximum=51, step=1, value=18)
                    enc_preset = gr.Dropdown(
                        label="Encoder preset",
                        choices=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                        value="p7",
                    )
                with gr.Row():
                    target_height = gr.Number(label="Target height (0=auto)", value=0)
                    deinterlace = gr.Dropdown(
                        label="Deinterlace",
                        choices=["auto", "on", "off"],
                        value="auto",
                    )

            # ── Multi-GPU Split (auto from endpoint) ──
            _default_wmax = max(_get_endpoint_workers_max(DEFAULT_ENDPOINT), 1)
            split_gpus = gr.Slider(
                label="Workers (auto from endpoint)",
                minimum=1, maximum=max(_default_wmax, 8), step=1, value=_default_wmax,
                info=f"1 = single worker, 2+ = split video into chunks. Endpoint max: {_default_wmax}",
            )

            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    low_priority = gr.Checkbox(
                        label="Low Priority",
                        value=False,
                        info="Won't spin up new workers, only uses idle capacity",
                    )
                webhook_url = gr.Textbox(
                    label="Webhook URL (optional)",
                    placeholder="https://your-server.com/callback",
                    info="RunPod POSTs job result here on completion. Leave blank to use polling.",
                )

            # ── Probe ──
            probe_btn = gr.Button("📊 Probe Selected Files", size="sm")
            probe_output = gr.Markdown(value="", label="Video Info")
            probe_btn.click(fn=probe_selected_files,
                           inputs=[files_input, endpoint_choice, split_gpus, no_itm],
                           outputs=[probe_output, dn, tile, crf, enc_preset, target_height, split_gpus])

            # ── Run ──
            with gr.Row():
                run_btn = gr.Button("🚀 Process", variant="primary", size="lg", scale=2)
                cancel_all_btn = gr.Button("🛑 Cancel All", variant="stop", size="lg", scale=1)

            # ── Log ──
            log_output = gr.Markdown(value="Ready. Select files and click Process.")

        # ════════════ RIGHT COLUMN ════════════
        with gr.Column(scale=2):

            # ── Health ──
            gr.Markdown("### Worker Status")
            health_output = gr.Markdown(value="Click Refresh to check endpoint status.")
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")

            # ── Jobs ──
            gr.Markdown("### Active Jobs")
            jobs_output = gr.Markdown(value="No active jobs.")
            refresh_jobs_btn = gr.Button("🔄 Refresh Jobs", size="sm")

            # ── Output Info ──
            gr.Markdown("### Latest Output")
            output_info_display = gr.Markdown(value="No completed jobs yet.")
            refresh_output_btn = gr.Button("🔄 Refresh Output Info", size="sm")

            # ── R2 Browser ──
            gr.Markdown("### R2 Storage")
            with gr.Row():
                r2_browse_btn = gr.Button("📁 Refresh", size="sm")
                download_all_btn = gr.Button("⬇️ Download All", size="sm")
            r2_file_list = gr.CheckboxGroup(label="R2 Files", choices=[], interactive=True)
            r2_browse_output = gr.Markdown(value="")
            with gr.Row():
                r2_download_btn = gr.Button("⬇️ Download Selected", size="sm")
                r2_delete_btn = gr.Button("🗑 Delete Selected", size="sm", variant="stop")
                r2_delete_all_btn = gr.Button("🗑 Delete All", size="sm", variant="stop")
            download_output = gr.Markdown(value="")

            # ── Management ──
            gr.Markdown("### Management")
            with gr.Row():
                purge_btn = gr.Button("🗑 Purge Queue", size="sm")
                retry_failed_btn = gr.Button("🔁 Retry Failed", size="sm")
            purge_output = gr.Textbox(label="", interactive=False, max_lines=1)

            with gr.Row():
                cancel_input = gr.Textbox(label="Job ID to cancel", scale=3)
                cancel_btn = gr.Button("Cancel", size="sm", scale=1)
            cancel_output = gr.Textbox(label="", interactive=False, max_lines=2)

    # ── Events ───────────────────────────────────────────────────────
    all_inputs = [
        files_input, endpoint_choice,
        no_face, no_itm, no_denoise, tile, batch,
        dn, face_strength, highlight_boost, temporal_smooth, itm_model, hdr_mode,
        crf, enc_preset, target_height, deinterlace,
        fp16, force_full_range, preserve_grain, grain_strength, no_trt,
        dec_workers, split_gpus, low_priority, webhook_url,
        r2_input_select,
    ]

    preserve_grain.change(
        fn=lambda x: gr.Slider(visible=x),
        inputs=[preserve_grain],
        outputs=[grain_strength],
    )
    endpoint_choice.change(
        fn=on_endpoint_change,
        inputs=[endpoint_choice],
        outputs=[split_gpus, endpoint_info],
    )
    refresh_gpu_btn.click(
        fn=refresh_gpu_availability,
        inputs=[endpoint_choice],
        outputs=[endpoint_info],
    )
    run_btn.click(
        fn=process_videos,
        inputs=all_inputs,
        outputs=[log_output, health_output, jobs_output, output_info_display],
    )
    refresh_btn.click(fn=do_refresh_health, inputs=[endpoint_choice],
                      outputs=[health_output])
    refresh_jobs_btn.click(fn=do_refresh_jobs, inputs=[endpoint_choice],
                           outputs=[jobs_output])
    refresh_output_btn.click(fn=format_output_info, inputs=[],
                              outputs=[output_info_display])
    purge_btn.click(fn=do_purge, inputs=[endpoint_choice], outputs=[purge_output])
    cancel_btn.click(fn=do_cancel_job, inputs=[endpoint_choice, cancel_input],
                     outputs=[cancel_output])
    cancel_all_btn.click(fn=do_cancel_all, inputs=[endpoint_choice],
                         outputs=[purge_output])
    retry_failed_btn.click(fn=do_retry_failed, inputs=[endpoint_choice],
                           outputs=[purge_output])
    r2_browse_btn.click(fn=do_browse_r2_interactive, inputs=[],
                        outputs=[r2_file_list, r2_browse_output])
    download_all_btn.click(fn=do_download_results, inputs=[endpoint_choice],
                           outputs=[download_output])
    r2_download_btn.click(fn=do_r2_download_selected, inputs=[r2_file_list],
                          outputs=[download_output])
    r2_delete_btn.click(fn=do_r2_delete_selected, inputs=[r2_file_list],
                        outputs=[download_output])
    r2_delete_all_btn.click(fn=do_r2_delete_all, inputs=[],
                            outputs=[download_output])

    # R2 Input tab
    r2_input_refresh_btn.click(fn=do_refresh_r2_inputs, inputs=[],
                               outputs=[r2_input_select, r2_input_status])
    r2_input_probe_btn.click(fn=do_probe_r2_inputs,
                             inputs=[r2_input_select, endpoint_choice, split_gpus, no_itm],
                             outputs=[r2_input_status, dn, tile, crf, enc_preset, target_height, split_gpus])
    r2_requeue_inputs = [
        r2_input_select, endpoint_choice,
        no_face, no_itm, tile, batch,
        dn, face_strength, itm_model, hdr_mode,
        crf, enc_preset, target_height, deinterlace,
        fp16, force_full_range, preserve_grain, grain_strength, no_trt,
        dec_workers, split_gpus,
        low_priority, webhook_url,
    ]
    r2_input_requeue_btn.click(fn=do_requeue_r2_inputs, inputs=r2_requeue_inputs,
                               outputs=[r2_input_status])

# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Endpoints: {[e.get('name') for e in ALL_ENDPOINTS]}")
    print(f"R2 Bucket: {BUCKET_NAME}")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Base(primary_hue=gr.themes.colors.blue),
    )
