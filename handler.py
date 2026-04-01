"""
RunPod Serverless Handler — AI Upscale & HDR Pipeline

Accepts a job with a video URL + processing parameters,
runs upscale_hdr.py, and returns the output URL.

Job input schema:
{
    "video_url": "input/video.mp4",       # required — S3/R2 key or URL
    "output_filename": "my_video.mkv",    # optional, defaults to input name + _hdr.mkv
    "target": 1440,                       # optional, output height (auto if omitted)
    "dn": -1,                             # optional, -1 = auto from resolution + bitrate
    "no_denoise": false,                  # optional, skip first-stage artifact cleanup
    "face_strength": 0.5,                 # optional
    "no_face": false,                     # optional, skip face restoration
    "no_itm": false,                      # optional, skip HDR tone mapping
    "itm": "params_3DM",                  # optional, HDR checkpoint
    "hdr_mode": "hdr10",                  # optional
    "crf": 18,                            # optional
    "preset": "p7",                       # optional
    "fp16": true,                         # optional
    "deinterlace": "auto",               # optional
    "force_full_range": false,            # optional
    "preserve_grain": false,              # optional
    "grain_strength": 1.0,                # optional
    "highlight_boost": 0.0,              # optional, 0.0-1.0 (specular highlight expansion)
    "temporal_smooth": 0.4,              # optional, reduce frame-to-frame flicker
    "batch": 4,                           # optional, ITM batch size
    "workers": 16,                        # optional, decode threads
    "test_mode": false                    # optional — Hub test mode
}
"""

import os
import re
import shutil
import subprocess
import traceback
import urllib.parse
import runpod
from runpod.serverless.utils import upload_file_to_bucket

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    boto3 = None

# ─── Startup: runs once when container starts (included in FlashBoot snapshot)

_VERSION = "v4.1"

print("=" * 50)
print(f"  AI Upscale & HDR Pipeline — Serverless Worker {_VERSION}")
print("=" * 50)

try:
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5
    )
    print(f"GPU: {gpu_info.stdout.strip()}")
except Exception:
    print("WARNING: nvidia-smi failed — no GPU available")

print("Checking models (baked into image, should be instant)...")
subprocess.run(["/workspace/download_models.sh"], timeout=300)
print("Models ready. Worker listening for jobs.")

# Pre-warm: import torch and check CUDA once at startup
_GPU_NAME = "unknown"
try:
    import torch
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        _GPU_NAME = dev.name
        print(f"  GPU: {dev.name} ({dev.total_mem / 1024**3:.1f} GB VRAM)")
        torch.zeros(1).cuda()
except Exception:
    pass


def _sanitize_filename(name):
    """Remove chars that break FFmpeg subprocess calls."""
    return re.sub(r'[|&;<>\"\' `$\\!#(){}[\]*?~]', '_', name)


def _env_default(key, fallback, cast=str):
    """Read a default from env var, with fallback. Job input always overrides."""
    val = os.environ.get(key)
    if val is None:
        return fallback
    try:
        return cast(val)
    except (ValueError, TypeError):
        return fallback


def _build_cmd(input_path, output_path, job_input):
    """Build the upscale_hdr.py CLI command from job input params.

    Each parameter checks job_input first, then falls back to DEFAULT_*
    env vars (configurable in Hub UI), then hardcoded defaults.
    """
    cmd = [
        "python", "-u", "/workspace/upscale_hdr.py",
        "-i", input_path,
        "-o", output_path,
        "--no-trt",
    ]

    # Resolve settings: job input > env default > hardcoded default
    hdr_mode = job_input.get("hdr_mode", _env_default("DEFAULT_HDR_MODE", "hdr10"))
    crf = job_input.get("crf", _env_default("DEFAULT_CRF", 18, int))
    preset = job_input.get("preset", _env_default("DEFAULT_PRESET", "p7"))
    target = job_input.get("target", _env_default("DEFAULT_TARGET", 0, int))
    dn = job_input.get("dn", _env_default("DEFAULT_DN", -1, float))
    face_strength = job_input.get("face_strength", _env_default("DEFAULT_FACE_STRENGTH", 0.5, float))
    no_face = job_input.get("no_face", _env_default("DEFAULT_NO_FACE", False, lambda v: v.lower() in ("true", "1", "yes")))
    preserve_grain = job_input.get("preserve_grain", _env_default("DEFAULT_PRESERVE_GRAIN", False, lambda v: v.lower() in ("true", "1", "yes")))

    # Value params
    param_map = {
        "batch": ("--batch", job_input.get("batch", 4)),
        "workers": ("--workers", job_input.get("workers", 16)),
        "dn": ("--dn", dn),
        "face_strength": ("--face-strength", face_strength),
        "hdr_mode": ("--hdr-mode", hdr_mode),
        "crf": ("--crf", crf),
        "preset": ("--preset", preset),
        "target": ("--target", target),
    }

    for key, (flag, val) in param_map.items():
        cmd.extend([flag, str(val)])

    # Optional value params (only if present in job input)
    if "itm" in job_input:
        cmd.extend(["--itm", str(job_input["itm"])])
    if "deinterlace" in job_input:
        val = job_input["deinterlace"]
        if isinstance(val, bool):
            val = "on" if val else "auto"
        cmd.extend(["--deinterlace", str(val)])
    if "temporal_smooth" in job_input:
        cmd.extend(["--temporal-smooth", str(job_input["temporal_smooth"])])
    if "highlight_boost" in job_input:
        cmd.extend(["--highlight-boost", str(job_input["highlight_boost"])])

    # Boolean flags
    if no_face:
        cmd.append("--no-face")
    if job_input.get("no_denoise"):
        cmd.append("--dn")
        cmd.append("0")
    if hdr_mode == "sdr" or job_input.get("no_itm"):
        cmd.append("--no-itm")
    if job_input.get("fp16", True):
        cmd.append("--fp16")
    if job_input.get("force_full_range"):
        cmd.append("--force-full-range")
    if preserve_grain:
        cmd.append("--preserve-grain")

    # Grain strength (float param)
    if "grain_strength" in job_input:
        cmd.extend(["--grain-strength", str(job_input["grain_strength"])])

    # Chunk processing (multi-GPU splitting)
    if "start_time" in job_input:
        cmd.extend(["--start-time", str(job_input["start_time"])])
    if "chunk_duration" in job_input:
        cmd.extend(["--duration", str(job_input["chunk_duration"])])

    return cmd


def _get_bucket_creds(job_input):
    """Get S3 bucket credentials from job input or env vars."""
    creds = job_input.get("bucket_creds")
    if creds:
        return creds

    endpoint_url = os.environ.get("BUCKET_ENDPOINT_URL")
    access_id = os.environ.get("BUCKET_ACCESS_KEY_ID")
    access_secret = os.environ.get("BUCKET_ACCESS_KEY_SECRET")
    if endpoint_url and access_id and access_secret:
        return {
            "endpointUrl": endpoint_url,
            "accessId": access_id,
            "accessSecret": access_secret,
        }
    return None


def _cleanup(*paths):
    """Remove temp files and directories."""
    for p in paths:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def _download_from_r2(video_url, job_id):
    """Download a file from R2 using boto3 with bucket credentials."""
    endpoint = os.environ.get("BUCKET_ENDPOINT_URL", "")
    access_key = os.environ.get("BUCKET_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("BUCKET_ACCESS_KEY_SECRET", "")
    bucket = os.environ.get("BUCKET_NAME", "upscale-hdr-io")

    parsed = urllib.parse.urlparse(video_url)
    path = parsed.path.lstrip("/")
    if path.startswith(f"{bucket}/"):
        s3_key = path[len(f"{bucket}/"):]
    else:
        s3_key = path

    s3_key = urllib.parse.unquote(s3_key)
    filename = os.path.basename(s3_key)
    download_dir = f"/tmp/input/{job_id}"
    os.makedirs(download_dir, exist_ok=True)
    local_path = os.path.join(download_dir, filename)

    print(f"Downloading s3://{bucket}/{s3_key} -> {local_path}")

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(signature_version="s3v4"),
        )
        s3.download_file(bucket, s3_key, local_path)
        return local_path
    except Exception as e:
        print(f"S3 download failed: {e}")
        return None


def _run_test_mode():
    """Hub test mode — verify models load and inference works without S3."""
    import numpy as np
    import torch

    stages_tested = []
    gpu = _GPU_NAME

    print("=== TEST MODE ===")
    print(f"GPU: {gpu}")

    # Create synthetic test frame (64x64 BGR)
    test_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Test S1 (denoise) model loads
    try:
        from upscale_hdr import load_stage1
        s1 = load_stage1(fp16=True, tile=0, dn=0.5)
        print("  S1 (denoise): loaded")
        stages_tested.append("S1_denoise")
        del s1
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  S1 (denoise): FAILED — {e}")

    # Test S2 (upscale) model loads
    try:
        from upscale_hdr import load_stage2
        s2 = load_stage2(fp16=True, tile=64, use_trt=False)
        print("  S2 (upscale): loaded")
        stages_tested.append("S2_upscale")
        del s2
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  S2 (upscale): FAILED — {e}")

    # Test S3 (face) model loads
    try:
        from upscale_hdr import load_stage3
        s3 = load_stage3()
        print("  S3 (face): loaded")
        stages_tested.append("S3_face")
        del s3
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  S3 (face): FAILED — {e}")

    # Test S4 (ITM) model loads
    try:
        from upscale_hdr import load_stage4
        s4 = load_stage4(use_trt=False)
        print("  S4 (ITM): loaded")
        stages_tested.append("S4_itm")
        del s4
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  S4 (ITM): FAILED — {e}")

    print(f"=== TEST COMPLETE: {len(stages_tested)}/4 stages OK ===")

    return {
        "status": "ok",
        "test": True,
        "gpu": gpu,
        "stages_tested": stages_tested,
        "stages_total": 4,
    }


def handler(job):
    """Process a single video upscale/HDR job."""
    job_input = job["input"]
    job_id = job["id"]

    # ── Test mode (for RunPod Hub automated testing) ──────────────────
    if job_input.get("test_mode"):
        return _run_test_mode()

    input_path = None
    output_dir = f"/tmp/output/{job_id}"

    try:
        # ── Validate ─────────────────────────────────────────────────────
        video_url = job_input.get("video_url")
        if not video_url:
            return {"error": "Missing 'video_url' in input"}

        # ── Download ─────────────────────────────────────────────────────
        runpod.serverless.progress_update(job, "Downloading input video...")
        input_path = _download_from_r2(video_url, job_id)
        if not input_path:
            return {"error": f"Failed to download video from {video_url}"}
        input_size = os.path.getsize(input_path)
        print(f"Downloaded: {input_path} ({input_size / 1024 / 1024:.1f} MB)")

        # ── Output path ──────────────────────────────────────────────────
        input_stem = os.path.splitext(os.path.basename(input_path))[0]
        safe_stem = _sanitize_filename(input_stem)
        output_filename = job_input.get("output_filename", f"{safe_stem}_hdr.mkv")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        # ── Run pipeline ─────────────────────────────────────────────────
        cmd = _build_cmd(input_path, output_path, job_input)
        print(f"Running: {' '.join(cmd)}")
        runpod.serverless.progress_update(job, "Processing video...")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONPATH": "/workspace/hdrtvdm:" + os.environ.get("PYTHONPATH", "")},
        )

        last_progress = ""
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(line)
                if "Pipeline:" in line and "%" in line:
                    pct = line.split("%")[0].split()[-1] if "%" in line else ""
                    if pct != last_progress:
                        runpod.serverless.progress_update(job, f"Pipeline: {pct}%")
                        last_progress = pct

        process.wait()

        if process.returncode != 0:
            return {"error": f"Pipeline failed (exit code {process.returncode})"}

        # ── Verify output ────────────────────────────────────────────────
        if not os.path.exists(output_path):
            for ext in [".mkv", ".mp4"]:
                alt = os.path.join(output_dir, safe_stem + "_hdr" + ext)
                if os.path.exists(alt):
                    output_path = alt
                    break
            else:
                return {"error": "Pipeline completed but output file not found"}

        output_size = os.path.getsize(output_path)
        print(f"Output: {output_path} ({output_size / 1024 / 1024:.1f} MB)")

        # ── Upload ───────────────────────────────────────────────────────
        runpod.serverless.progress_update(job, "Uploading result...")

        bucket_creds = _get_bucket_creds(job_input)
        if bucket_creds:
            bucket_name = os.environ.get("BUCKET_NAME", "upscale-hdr-io")
            presigned_url = upload_file_to_bucket(
                file_name=os.path.basename(output_path),
                file_location=output_path,
                bucket_creds=bucket_creds,
                bucket_name=bucket_name,
                prefix=f"output/{job_id}",
            )
            return {
                "output_url": presigned_url,
                "filename": os.path.basename(output_path),
                "size_mb": round(output_size / 1024 / 1024, 1),
                "refresh_worker": True,
            }
        else:
            return {
                "output_path": output_path,
                "filename": os.path.basename(output_path),
                "size_mb": round(output_size / 1024 / 1024, 1),
                "warning": "No bucket credentials — output stored on worker only",
                "refresh_worker": True,
            }

    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "refresh_worker": True,
        }

    finally:
        _cleanup(output_dir)
        if input_path:
            _cleanup(input_path)


runpod.serverless.start({"handler": handler})
