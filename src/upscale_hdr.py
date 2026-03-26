"""
AI Upscale & HDR Pipeline — upscale_hdr.py

Four-stage pipeline: artifact cleanup, upscale, face restoration, HDR tone mapping.
Designed for Windows 10/11 with NVIDIA RTX GPU (CUDA + NVENC).

HDR output modes:
  - HDR10 (static metadata)  — libx265, MaxCLL/MaxFALL from frame analysis
  - HDR10+ (dynamic metadata) — libx265 software encode, per-frame luminance bezier curves
"""

import argparse
import re
import subprocess
import json
import queue
import threading
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch
import cv2
from tqdm import tqdm
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer

# ---------------------------------------------------------------------------
# Resolve model directory relative to this script
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _SCRIPT_DIR / "models"


# ═══════════════════════════════════════════════════════════════════════════
#  PROBING & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def probe_video(path):
    """Returns dict: width, height, fps, codec, bitrate_kbps, duration, color metadata."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    data = json.loads(subprocess.check_output(cmd))
    vs = next(s for s in data["streams"] if s["codec_type"] == "video")
    # Parse FPS — prefer r_frame_rate but fall back to avg_frame_rate
    # for VFR/mjpeg containers where r_frame_rate is the timebase (e.g. 90000/1)
    n, d = 30, 1
    for fps_key in ("r_frame_rate", "avg_frame_rate"):
        try:
            _n, _d = map(int, vs.get(fps_key, "0/1").split("/"))
            candidate = _n / _d if _d > 0 else 0
            if 1 <= candidate <= 240:  # sane FPS range
                n, d = _n, _d
                break
        except (ValueError, ZeroDivisionError):
            continue

    # Bitrate: try stream-level first, fall back to format-level
    bitrate = vs.get("bit_rate")
    if not bitrate:
        bitrate = data.get("format", {}).get("bit_rate")
    bitrate_kbps = int(bitrate) // 1000 if bitrate else None

    duration = float(data.get("format", {}).get("duration", 0))

    # Color metadata
    color_range = vs.get("color_range")          # 'tv', 'pc', or None
    color_transfer = vs.get("color_transfer")     # 'bt709', 'smpte2084', etc.
    color_primaries = vs.get("color_primaries")   # 'bt709', 'bt2020', etc.

    # Treat unsignaled range as limited — per ITU-T H.264/H.265 spec default
    is_full_range = (color_range == "pc")

    # Detect HDR source — ITM on HDR input is wrong
    is_hdr = color_transfer in ("smpte2084", "arib-std-b67", "bt2020-10", "bt2020-12")

    return {
        "width": int(vs["width"]),
        "height": int(vs["height"]),
        "fps": n / d if d else 30.0,
        "codec": vs["codec_name"],
        "bitrate_kbps": bitrate_kbps,
        "duration": duration,
        "color_range": color_range,
        "color_transfer": color_transfer,
        "color_primaries": color_primaries,
        "is_full_range": is_full_range,
        "is_hdr": is_hdr,
    }


def detect_interlace(path, sample_frames=200):
    """Returns True if source appears interlaced.

    First checks ffprobe field_order (reliable). Falls back to idet filter
    only if field_order is missing or ambiguous.
    """
    # Fast check: ffprobe field_order
    try:
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=field_order",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ]
        field_order = subprocess.run(
            probe_cmd, capture_output=True, text=True, timeout=10,
        ).stdout.strip().lower()
        if field_order == "progressive":
            return False
        if field_order in ("tt", "bb", "tb", "bt"):
            return True
    except Exception:
        pass

    # Fallback: idet filter analysis
    cmd = [
        "ffmpeg", "-i", str(path), "-vf", "idet",
        "-frames:v", str(sample_frames), "-an", "-f", "null", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stderr
    return "TFF:" in out and "TFF: 0 " not in out


# ═══════════════════════════════════════════════════════════════════════════
#  COLOR RANGE HANDLING
# ═══════════════════════════════════════════════════════════════════════════

# ANSI colors for terminal warnings
_WARN  = "\033[93m"   # yellow
_ERR   = "\033[91m"   # red
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def validate_color_range(probe_info, force_full_range=False):
    """
    Central validation step. Called once per file before any processing begins.

    Resolves the effective color range from probe metadata and the user's
    --force-full-range flag, prints appropriate warnings, and returns the
    final is_full_range bool to use for the entire file.
    """
    signaled_range  = probe_info.get("color_range")      # 'tv', 'pc', or None
    color_transfer  = probe_info.get("color_transfer")
    color_primaries = probe_info.get("color_primaries")
    is_hdr          = probe_info.get("is_hdr", False)

    print(f"\n{'─'*60}")
    print(f"  Color Range Validation")
    print(f"{'─'*60}")
    print(f"  Signaled color_range:    {signaled_range or 'not present in file'}")
    print(f"  color_transfer:          {color_transfer or 'not present'}")
    print(f"  color_primaries:         {color_primaries or 'not present'}")
    print(f"  --force-full-range:      {force_full_range}")
    print(f"{'─'*60}")

    # HDR source check
    if is_hdr:
        print(f"\n  {_ERR}{_BOLD}[ERROR] Input is already HDR.{_RESET}")
        print(f"  {_ERR}  color_transfer = {color_transfer}{_RESET}")
        print(f"  {_ERR}  Running the ITM stage on HDR input is wrong —{_RESET}")
        print(f"  {_ERR}  HDRTVDM was trained to map SDR → HDR.{_RESET}")
        print(f"  {_ERR}  Feeding it HDR will produce clipped, over-expanded output.{_RESET}")
        print(f"\n  {_BOLD}Fix: add --no-itm to your command.{_RESET}")
        print(f"  The upscale and face restoration stages still run normally.\n")

    # Mismatch: file says tv, user forcing pc
    if signaled_range == "tv" and force_full_range:
        print(f"\n  {_WARN}{_BOLD}[WARN] --force-full-range is set but the file "
              f"metadata signals limited range (tv).{_RESET}")
        print(f"  {_WARN}  Treating this file as full range when it is not will cause:{_RESET}")
        print(f"  {_WARN}    • Blacks crushed to below-black (illegal values in output){_RESET}")
        print(f"  {_WARN}    • Highlights clipped — detail above 235 discarded{_RESET}")
        print(f"  {_WARN}    • HDRTVDM tone expansion miscalibrated{_RESET}")
        print(f"\n  {_WARN}  You probably do NOT want --force-full-range for this file.{_RESET}")
        print(f"  {_WARN}  Remove it unless you are certain the metadata is wrong.{_RESET}\n")
        return True   # honour user's explicit override but warn loudly

    # File says pc, user did NOT pass --force-full-range — auto-enable
    if signaled_range == "pc" and not force_full_range:
        print(f"\n  {_WARN}{_BOLD}[AUTO] File metadata signals full range (pc) "
              f"— auto-enabling full range processing.{_RESET}")
        print(f"  {_WARN}  normalize_frame will map 0–255 → 0.0–1.0{_RESET}")
        print(f"  {_WARN}  If output looks wrong, check if metadata is accurate.{_RESET}\n")
        return True

    # Ambiguous: range not signaled in file
    if signaled_range is None and not force_full_range:
        print(f"\n  {_WARN}[WARN] color_range is not signaled in this file's metadata.{_RESET}")
        print(f"  {_WARN}  Spec default (ITU-T H.264/H.265) is limited range (tv).{_RESET}")
        print(f"  {_WARN}  Proceeding as limited range. If output blacks look lifted{_RESET}")
        print(f"  {_WARN}  or whites clip, re-run with --force-full-range.{_RESET}\n")
        return False

    # All clear
    effective = force_full_range or (signaled_range == "pc")
    if effective:
        print(f"\n  [INFO] Full range (pc) confirmed. normalize_frame will map 0–255 → 0.0–1.0\n")
    else:
        print(f"\n  [INFO] Limited range (tv) confirmed. normalize_frame will map 16–235 → 0.0–1.0\n")

    return effective


def check_first_frame_range(path, threshold_low=8, threshold_high=245):
    """
    Heuristic pixel-level check for files with unsignaled color_range.
    Decodes the first frame and checks whether significant pixel mass
    exists below 16 or above 235 — if so, the file is almost certainly
    full range regardless of missing metadata.
    """
    cmd = [
        "ffmpeg", "-i", str(path), "-vframes", "1",
        "-f", "rawvideo", "-pix_fmt", "gray", "-v", "quiet", "-",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if not result.stdout:
        return

    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    total = raw.size
    if total == 0:
        return

    pct_very_dark = np.sum(raw < threshold_low) / total * 100
    pct_very_bright = np.sum(raw > threshold_high) / total * 100

    if pct_very_dark > 0.5 or pct_very_bright > 0.5:
        print(f"  {_WARN}[WARN] Pixel heuristic suggests this may be full range:{_RESET}")
        print(f"  {_WARN}    {pct_very_dark:.2f}% of pixels below {threshold_low} "
              f"(below limited-range black){_RESET}")
        print(f"  {_WARN}    {pct_very_bright:.2f}% of pixels above {threshold_high} "
              f"(above limited-range white){_RESET}")
        print(f"  {_WARN}  If output looks wrong, re-run with --force-full-range.{_RESET}\n")


def normalize_frame_cuda(frame_tensor, is_full_range):
    """GPU-accelerated frame normalization. Input: uint8 tensor on CUDA, Output: float32 tensor on CUDA."""
    frame = frame_tensor.float()
    if is_full_range:
        return frame / 255.0
    else:
        frame = frame.clamp(16.0, 235.0)
        return (frame - 16.0) / (235.0 - 16.0)


def normalize_frame(frame_bgr, is_full_range):
    """
    Converts a raw BGR uint8 frame to the correct float range for AI inference.

    Limited range (tv): 16–235 → 0.0–1.0
    Full range (pc):    0–255 → 0.0–1.0
    """
    frame = frame_bgr.astype(np.float32)
    if is_full_range:
        return frame / 255.0
    else:
        frame = np.clip(frame, 16.0, 235.0)
        return (frame - 16.0) / (235.0 - 16.0)


def denormalize_frame(frame_float, is_full_range):
    """
    Converts AI output [0.0, 1.0] float back to uint8 BGR for the encoder pipe.
    Output is ALWAYS limited range (tv) — HDR10 spec requires it.
    """
    frame = np.clip(frame_float, 0.0, 1.0)
    out = frame * (235.0 - 16.0) + 16.0
    return out.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
#  BITRATE-BASED DENOISING RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════

def recommend_dn(height, bitrate_kbps):
    """
    Recommend denoising strength based on resolution + bitrate.

    Logic: lower bitrate relative to resolution = more compression artifacts.
    Returns (recommended_dn, explanation_string).
    """
    if bitrate_kbps is None:
        return 0.5, "Bitrate unknown -- using default 0.5"

    if height <= 480:
        if bitrate_kbps < 1500:
            dn, desc = 0.85, "Very low bitrate 480p -- heavy artifact cleanup"
        elif bitrate_kbps < 3000:
            dn, desc = 0.7, "Low bitrate 480p -- strong artifact cleanup"
        elif bitrate_kbps < 5000:
            dn, desc = 0.5, "Moderate bitrate 480p -- balanced cleanup"
        else:
            dn, desc = 0.35, "Decent bitrate 480p -- light cleanup"
    elif height <= 720:
        if bitrate_kbps < 2000:
            dn, desc = 0.75, "Very low bitrate 720p -- strong artifact cleanup"
        elif bitrate_kbps < 4000:
            dn, desc = 0.6, "Low bitrate 720p -- moderate-strong cleanup"
        elif bitrate_kbps < 6000:
            dn, desc = 0.45, "Moderate bitrate 720p -- balanced cleanup"
        else:
            dn, desc = 0.3, "Good bitrate 720p -- light cleanup"
    else:  # 1080p+
        if bitrate_kbps < 3000:
            dn, desc = 0.7, "Very low bitrate 1080p -- strong artifact cleanup"
        elif bitrate_kbps < 6000:
            dn, desc = 0.5, "Low bitrate 1080p -- moderate cleanup"
        elif bitrate_kbps < 10000:
            dn, desc = 0.35, "Moderate bitrate 1080p -- light cleanup"
        elif bitrate_kbps < 15000:
            dn, desc = 0.25, "Good bitrate 1080p -- minimal cleanup"
        else:
            dn, desc = 0.15, "High bitrate 1080p -- very light cleanup"

    return dn, f"{desc} ({bitrate_kbps} kbps @ {height}p)"


# ═══════════════════════════════════════════════════════════════════════════
#  FACE AUTO-DETECTION (lightweight check on a few sample frames)
# ═══════════════════════════════════════════════════════════════════════════

def detect_faces_in_video(path, sample_count=5, duration=None, width=None, height=None):
    """
    Extract a few sample frames and run OpenCV face detection.
    Returns (has_faces: bool, face_count: int, description: str).
    Uses the fast Haar cascade -- no GPU needed for this check.

    width/height: pass from probe_video to avoid re-probing per frame.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if duration is None or duration <= 0:
        duration = 60  # assume at least 60s

    # Probe once for dimensions if not provided
    if width is None or height is None:
        probe_info = probe_video(path)
        width, height = probe_info["width"], probe_info["height"]

    # Sample frames spread across the video
    timestamps = []
    if duration > 10:
        step = duration / (sample_count + 1)
        timestamps = [step * (i + 1) for i in range(sample_count)]
    else:
        timestamps = [duration / 2]

    total_faces = 0
    frames_with_faces = 0
    expected = width * height * 3

    for ts in timestamps:
        cmd = [
            "ffmpeg", "-ss", f"{ts:.2f}", "-i", str(path),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "quiet", "-",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode != 0 or not result.stdout:
                continue
        except (subprocess.TimeoutExpired, Exception):
            continue

        raw = result.stdout
        if len(raw) < expected:
            continue

        frame = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Scale down for faster detection
        scale = min(1.0, 640 / width)
        if scale < 1.0:
            small = cv2.resize(gray, None, fx=scale, fy=scale)
        else:
            small = gray

        faces = face_cascade.detectMultiScale(small, 1.1, 5, minSize=(30, 30))
        n = len(faces)
        total_faces += n
        if n > 0:
            frames_with_faces += 1

    checked = len(timestamps)
    if checked == 0:
        return False, 0, "Could not extract sample frames for face detection"

    has_faces = frames_with_faces > 0
    pct = (frames_with_faces / checked) * 100

    if not has_faces:
        desc = f"No faces detected in {checked} sample frames -- consider --no-face"
    elif frames_with_faces == checked:
        desc = f"Faces in all {checked} samples ({total_faces} total) -- face restoration recommended"
    else:
        desc = f"Faces in {frames_with_faces}/{checked} samples ({pct:.0f}%) -- face restoration may help"

    return has_faces, total_faces, desc


# ═══════════════════════════════════════════════════════════════════════════
#  SCALE LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def get_scale(height, target_override=None, bitrate_kbps=0, codec="", width=0):
    """
    Adaptive outscale based on input analysis.  Max output = 4K (2160p).

    Decision factors:
      - Source resolution: how far to upscale
      - Source bitrate per pixel: quality indicator (higher = cleaner source)
      - Codec: modern codecs (h265/av1/vp9) are more efficient at same bitrate

    Returns (scale_factor, target_height).
    """
    if target_override and target_override > 0:
        target = min(target_override, 2160)  # hard cap at 4K
        return target / height, target

    # Modern codecs are ~40% more efficient, so adjust perceived quality
    efficient_codecs = {"hevc", "h265", "av1", "vp9"}
    codec_bonus = 1.4 if codec.lower() in efficient_codecs else 1.0
    effective_bitrate = (bitrate_kbps or 0) * codec_bonus

    # Bits per pixel per frame (quality density metric)
    # Higher = cleaner source = can push higher upscale
    # Use actual width if provided, otherwise estimate from 16:9
    actual_w = width if width > 0 else int(height * 16 / 9)
    pixels = height * actual_w
    bpp = (effective_bitrate * 1000) / (pixels * 30) if pixels > 0 else 0

    if height <= 360:
        # 240p-360p: very low res
        # Clean source → 1080p, dirty → 720p
        target = 1080 if bpp > 0.04 else 720
    elif height <= 480:
        # 480p (DVD-era)
        # Clean source → 2160p, moderate → 1080p
        if bpp > 0.08:
            target = 2160
        elif bpp > 0.03:
            target = 1440
        else:
            target = 1080
    elif height <= 720:
        # 720p
        # Clean source → 2160p, moderate → 1440p, dirty → 1080p
        if bpp > 0.06:
            target = 2160
        elif bpp > 0.03:
            target = 1440
        else:
            target = 1080
    elif height <= 1080:
        # 1080p → always 4K (2x is the sweet spot for Real-ESRGAN)
        target = 2160
    elif height <= 1440:
        # 1440p → 4K (minor upscale)
        target = 2160
    else:
        # Already 4K+ → no upscale, just process (HDR/denoise/face)
        target = height

    target = min(target, 2160)  # hard cap at 4K
    scale = target / height
    # Don't downscale
    if scale < 1.0:
        return 1.0, height
    return scale, target


# ═══════════════════════════════════════════════════════════════════════════
#  HDR LUMINANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

# PQ (SMPTE ST 2084) constants
_PQ_M1 = 0.1593017578125
_PQ_M2 = 78.84375
_PQ_C1 = 0.8359375
_PQ_C2 = 18.8515625
_PQ_C3 = 18.6875
_PQ_PEAK_NITS = 10000.0


def pq_to_nits(pq_value):
    """Convert PQ-encoded value [0,1] to linear luminance in nits [0,10000]."""
    if pq_value <= 0:
        return 0.0
    vp = pq_value ** (1.0 / _PQ_M2)
    num = max(vp - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * vp
    if den <= 0:
        return 0.0
    return _PQ_PEAK_NITS * (num / den) ** (1.0 / _PQ_M1)


def analyze_hdr_frame(hdr_frame_u16):
    """
    Analyze an HDR10 frame (RGB uint16 [0,65535]) for luminance statistics.

    Returns dict with:
      - max_nits:  peak luminance in nits (for MaxCLL)
      - avg_nits:  mean luminance in nits (for MaxFALL)
      - percentiles: dict of luminance percentiles for HDR10+ bezier curves
    """
    # Convert uint16 [0,65535] to float [0,1] PQ signal
    frame_f = hdr_frame_u16.astype(np.float32) / 65535.0

    # BT.2020 luminance coefficients (Y from RGB)
    # Y = 0.2627*R + 0.6780*G + 0.0593*B
    luminance_pq = (
        0.2627 * frame_f[:, :, 0]
        + 0.6780 * frame_f[:, :, 1]
        + 0.0593 * frame_f[:, :, 2]
    )

    # Convert PQ luminance to nits (vectorized for speed)
    # Use a lookup table approach for large frames
    lut_size = 4096
    lut_pq = np.linspace(0, 1, lut_size, dtype=np.float32)
    lut_nits = np.zeros(lut_size, dtype=np.float32)
    for i in range(lut_size):
        lut_nits[i] = pq_to_nits(lut_pq[i])

    # Map PQ values to LUT indices
    indices = np.clip((luminance_pq * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
    nits = lut_nits[indices]

    max_nits = float(np.max(nits))
    avg_nits = float(np.mean(nits))

    # Percentiles for HDR10+ tone mapping curves
    percentiles = {
        1: float(np.percentile(nits, 1)),
        5: float(np.percentile(nits, 5)),
        10: float(np.percentile(nits, 10)),
        25: float(np.percentile(nits, 25)),
        50: float(np.percentile(nits, 50)),
        75: float(np.percentile(nits, 75)),
        90: float(np.percentile(nits, 90)),
        95: float(np.percentile(nits, 95)),
        99: float(np.percentile(nits, 99)),
    }

    return {
        "max_nits": max_nits,
        "avg_nits": avg_nits,
        "percentiles": percentiles,
    }


class HDRMetadataCollector:
    """
    Collects per-frame luminance statistics during pipeline processing.
    Used for both static HDR10 (MaxCLL/MaxFALL) and HDR10+ (per-scene curves).
    """

    def __init__(self):
        self.frame_stats = []  # list of per-frame dicts
        self.global_max_nits = 0.0
        self.global_avg_nits_sum = 0.0
        self.frame_count = 0

    def add_frame(self, hdr_frame_u16):
        """Analyze a frame and store its stats."""
        stats = analyze_hdr_frame(hdr_frame_u16)
        self.frame_stats.append(stats)
        self.global_max_nits = max(self.global_max_nits, stats["max_nits"])
        self.global_avg_nits_sum += stats["avg_nits"]
        self.frame_count += 1
        return stats

    @property
    def max_cll(self):
        """MaxCLL: maximum content light level (peak nit across all frames)."""
        return int(min(self.global_max_nits, 10000))

    @property
    def max_fall(self):
        """MaxFALL: maximum frame-average light level."""
        if self.frame_count == 0:
            return 0
        return int(min(self.global_avg_nits_sum / self.frame_count, 10000))

    def generate_hdr10plus_json(self):
        """
        Generate HDR10+ metadata JSON (per-frame scene info).

        Format follows the Samsung HDR10+ specification:
        - targeted_system_display_maximum_luminance
        - maxscl (per-frame max RGB channel nits)
        - distribution_maxrgb (percentile curve)
        - bezier_curve_anchors (tone mapping guidance)
        """
        scenes = []
        for i, stats in enumerate(self.frame_stats):
            p = stats["percentiles"]

            # MaxSCL: max nits per color channel (we use overall max as approx)
            max_scl = [
                int(stats["max_nits"] * 10),  # 0.0001 cd/m2 units (x10)
                int(stats["max_nits"] * 10),
                int(stats["max_nits"] * 10),
            ]

            # Distribution: percentile values for the maxRGB distribution
            distribution = [
                {"percentage": 1, "percentile": int(p[1] * 10)},
                {"percentage": 5, "percentile": int(p[5] * 10)},
                {"percentage": 10, "percentile": int(p[10] * 10)},
                {"percentage": 25, "percentile": int(p[25] * 10)},
                {"percentage": 50, "percentile": int(p[50] * 10)},
                {"percentage": 75, "percentile": int(p[75] * 10)},
                {"percentage": 90, "percentile": int(p[90] * 10)},
                {"percentage": 95, "percentile": int(p[95] * 10)},
                {"percentage": 99, "percentile": int(p[99] * 10)},
            ]

            # Bezier curve anchors for tone mapping
            # Normalize percentile values to [0, 1] range relative to peak
            peak = max(stats["max_nits"], 1.0)
            anchors = [
                int((p[10] / peak) * 1023),
                int((p[25] / peak) * 1023),
                int((p[50] / peak) * 1023),
                int((p[75] / peak) * 1023),
                int((p[90] / peak) * 1023),
            ]
            # Clamp to valid range
            anchors = [max(0, min(1023, a)) for a in anchors]

            scene = {
                "BezierCurveData": {
                    "TargetedSystemDisplayMaximumLuminance": 400,
                    "KneePointX": int((p[90] / peak) * 1023) if peak > 0 else 512,
                    "KneePointY": int((p[90] / peak) * 1023) if peak > 0 else 512,
                    "NumberOfAnchors": len(anchors),
                    "Anchors": anchors,
                },
                "LuminanceParameters": {
                    "AverageRGB": int(stats["avg_nits"] * 10),
                    "MaxSCL": max_scl,
                    "DistributionMaxRGB": distribution,
                },
            }
            scenes.append(scene)

        metadata = {
            "HDR10PlusProfile": "A",
            "Version": "1.0",
            "TargetedSystemDisplayMaximumLuminance": 400,
            "SceneInfo": scenes,
            "SceneInfoCount": len(scenes),
        }
        return metadata

    def print_summary(self):
        """Print HDR metadata summary."""
        print(f"\n  [HDR METADATA]")
        print(f"    MaxCLL  : {self.max_cll} nits")
        print(f"    MaxFALL : {self.max_fall} nits")
        print(f"    Frames analyzed: {self.frame_count}")
        if self.frame_count > 0:
            peak_frame = max(self.frame_stats, key=lambda s: s["max_nits"])
            idx = self.frame_stats.index(peak_frame)
            print(f"    Peak frame: #{idx} ({peak_frame['max_nits']:.0f} nits)")


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def _try_compile(model, name="model"):
    """Apply safe optimizations: TF32 + channels_last.

    torch.compile is disabled for HDRTVDM's LSN architecture (dynamic shapes
    cause BackendCompilerFailed). TF32 alone gives ~3x faster float32 matmul
    on Ampere+ GPUs with no compatibility risk.
    """
    torch.set_float32_matmul_precision('high')
    print(f"    [OPT] {name}: TF32 matmul + channels_last enabled")
    return model




def load_stage1(model_path=None, fp16=True, tile=0, dn=0.5):
    """
    Stage 1: realesr-general-x4v3 — same-res artifact cleanup.

    When dn > 0 and the WDN model is available, loads both models for
    denoising interpolation. dn=0 keeps grain/noise, dn=1 smooths aggressively.
    """
    if model_path is None:
        model_path = str(_MODEL_DIR / "realesrgan" / "realesr-general-x4v3.pth")
    model = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_conv=32, upscale=4, act_type="prelu",
    )
    # channels_last for better GPU memory access patterns
    model = model.to(memory_format=torch.channels_last)

    wdn_path = Path(model_path).parent / "realesr-general-wdn-x4v3.pth"
    if dn > 0 and wdn_path.exists():
        return RealESRGANer(
            scale=4, model_path=[model_path, str(wdn_path)],
            model=model, tile=tile, tile_pad=10, half=fp16,
            dni_weight=[1 - dn, dn],
        )

    return RealESRGANer(
        scale=4, model_path=model_path,
        model=model, tile=tile, tile_pad=10, half=fp16,
    )


def load_stage2(model_path=None, fp16=True, tile=0, use_trt=False):
    """Stage 2: RealESRGAN_x4plus — adaptive outscale upscale."""
    if model_path is None:
        model_path = str(_MODEL_DIR / "realesrgan" / "RealESRGAN_x4plus.pth")
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=4,
    )
    model = model.to(memory_format=torch.channels_last)
    upscaler = RealESRGANer(
        scale=4, model_path=model_path,
        model=model, tile=tile, tile_pad=10, half=fp16,
    )

    return upscaler


def load_stage3(model_path=None):
    """Stage 3: GFPGANv1.4 — face restoration, no resize."""
    if model_path is None:
        model_path = str(_MODEL_DIR / "gfpgan" / "GFPGANv1.4.pth")
    return GFPGANer(
        model_path=model_path, upscale=1,
        arch="clean", channel_multiplier=2,
    )


def load_stage4(checkpoint_name="params_3DM", use_trt=False):
    """Stage 4: HDRTVDM LSN — SDR -> HDR10 inverse tone mapping."""
    checkpoint_path = _MODEL_DIR / "hdrtvdm" / f"{checkpoint_name}.pth"
    sys.path.insert(0, str(_MODEL_DIR / "hdrtvdm"))
    sys.path.insert(0, "/workspace/hdrtvdm")
    from models_hdrtvdm import LSN  # noqa: E402
    m = LSN().cuda().eval()
    m.load_state_dict(torch.load(str(checkpoint_path), map_location="cuda", weights_only=True))
    # channels_last + torch.compile for ITM model (~2x speedup)
    m = m.to(memory_format=torch.channels_last)
    m = _try_compile(m, "HDRTVDM")

    return m


# ═══════════════════════════════════════════════════════════════════════════
#  GRAIN ANALYSIS & SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_grain(frame_bgr, sample_size=256):
    """Estimate film grain characteristics from a source frame.

    Samples patches from the frame, applies a Gaussian blur to isolate the
    noise layer, then measures standard deviation per channel (intensity)
    and spatial frequency (coarseness).

    Returns dict: {intensity_per_channel: [B,G,R], coarseness: float}
    """
    h, w = frame_bgr.shape[:2]
    frame_f = frame_bgr.astype(np.float32)

    # Sample up to 4 random patches to average grain characteristics
    patches = []
    for _ in range(4):
        y = np.random.randint(0, max(1, h - sample_size))
        x = np.random.randint(0, max(1, w - sample_size))
        patch = frame_f[y:y+sample_size, x:x+sample_size]
        patches.append(patch)

    intensities = []
    for patch in patches:
        smoothed = cv2.GaussianBlur(patch, (5, 5), 1.5)
        noise = patch - smoothed
        # Per-channel std dev
        intensities.append([noise[:, :, c].std() for c in range(3)])

    avg_intensity = np.mean(intensities, axis=0).tolist()

    # Coarseness: ratio of noise after larger blur vs smaller blur
    # Higher = coarser grain
    patch = patches[0]
    small_blur = cv2.GaussianBlur(patch, (3, 3), 0.8)
    large_blur = cv2.GaussianBlur(patch, (7, 7), 2.5)
    fine_noise = (patch - small_blur).std()
    coarse_noise = (patch - large_blur).std()
    coarseness = float(coarse_noise / max(fine_noise, 1e-6))

    return {
        "intensity": avg_intensity,  # [B, G, R] std devs
        "coarseness": coarseness,
    }


def synthesize_grain(frame, grain_profile, strength=1.0):
    """Apply synthetic film grain matching the analyzed profile.

    Works with both 8-bit (uint8) and 16-bit (uint16) HDR frames.
    Uses GPU for noise generation and blending on 4K+ frames.

    Args:
        frame: numpy array (H, W, 3), uint8 or uint16
        grain_profile: dict from analyze_grain()
        strength: 0.0 = no grain, 1.0 = match original, >1.0 = amplify
    """
    if strength <= 0 or grain_profile is None:
        return frame

    h, w, c = frame.shape
    is_16bit = frame.dtype == np.uint16
    intensity = grain_profile["intensity"]  # [B, G, R]
    coarseness = grain_profile["coarseness"]

    # Scale intensity for bit depth
    depth_scale = 257.0 if is_16bit else 1.0  # 65535/255 ≈ 257
    max_val = 65535.0 if is_16bit else 255.0

    # GPU path for large frames (4K = 8M+ pixels), CPU for small frames
    use_gpu = torch.cuda.is_available() and (h * w) > 2_000_000

    if use_gpu:
        # Generate noise on GPU — avoids large CPU→GPU transfer
        noise = torch.randn(h, w, c, device="cuda", dtype=torch.float32)

        # Apply coarseness via blur (coarser grain = more blur)
        if coarseness > 1.2:
            ksize = 5 if coarseness > 1.8 else 3
            sigma = min(coarseness * 0.6, 2.5)
            # Use torchvision gaussian_blur (N,C,H,W format)
            from torchvision.transforms.functional import gaussian_blur as tv_blur
            noise_nchw = noise.permute(2, 0, 1).unsqueeze(0)
            noise_nchw = tv_blur(noise_nchw, [ksize, ksize], [sigma, sigma])
            noise = noise_nchw.squeeze(0).permute(1, 2, 0)
            # Re-normalize per channel
            for ch in range(c):
                ch_std = noise[:, :, ch].std()
                if ch_std > 1e-6:
                    noise[:, :, ch] /= ch_std

        # Scale per-channel
        scale_t = torch.tensor(
            [i * depth_scale * strength for i in intensity],
            device="cuda", dtype=torch.float32,
        )
        noise *= scale_t

        # Add grain on GPU
        frame_t = torch.from_numpy(frame.astype(np.int32) if is_16bit else frame).cuda().float()
        result = (frame_t + noise).clamp_(0, max_val)
        out_dtype = np.uint16 if is_16bit else np.uint8
        return result.cpu().to(torch.int32).numpy().astype(out_dtype)
    else:
        # CPU path for small frames
        noise = np.random.randn(h, w, c).astype(np.float32)
        if coarseness > 1.2:
            ksize = 5 if coarseness > 1.8 else 3
            sigma = min(coarseness * 0.6, 2.5)
            noise = cv2.GaussianBlur(noise, (ksize, ksize), sigma)
            for ch in range(c):
                ch_std = noise[:, :, ch].std()
                if ch_std > 1e-6:
                    noise[:, :, ch] /= ch_std
        for ch in range(c):
            noise[:, :, ch] *= intensity[ch] * depth_scale * strength
        result = frame.astype(np.float32) + noise
        return np.clip(result, 0, max_val).astype(np.uint16 if is_16bit else np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def run_stage1(s1, frame_bgr, dn=0.5):
    """Stage 1: artifact cleanup at same resolution.

    dn is applied by updating dni_weight before enhance() — this allows
    changing denoising strength per-frame without reloading the model.
    The model is 4x but outscale=1 handles the downscale after inference.
    """
    import cv2
    h, w = frame_bgr.shape[:2]
    s1.scale = 4
    if hasattr(s1, "dni_weight") and s1.dni_weight is not None:
        s1.dni_weight = [1 - dn, dn]
    cleaned, _ = s1.enhance(frame_bgr, outscale=1)
    if cleaned.shape[1] != w or cleaned.shape[0] != h:
        cleaned = cv2.resize(cleaned, (w, h), interpolation=cv2.INTER_LINEAR)
    return cleaned


def run_stage2(s2, frame_bgr, outscale):
    """Stage 2: upscale the cleaned frame."""
    import cv2
    s2.scale = 4
    h, w = frame_bgr.shape[:2]
    target_w = int(w * outscale)
    target_h = int(h * outscale)
    target_w = target_w - (target_w % 2)
    target_h = target_h - (target_h % 2)
    upscaled, _ = s2.enhance(frame_bgr, outscale=outscale)
    if upscaled.shape[1] != target_w or upscaled.shape[0] != target_h:
        upscaled = cv2.resize(upscaled, (target_w, target_h),
                              interpolation=cv2.INTER_LINEAR)
    return upscaled


def run_fused_s1_s2(s1, s2, frame_bgr, outscale, dn=0.5):
    """Fused S1+S2: denoise at 4x then downscale to target — skips the
    intermediate downscale-to-1x + re-upscale-to-4x round-trip.

    When both S1 and S2 are needed, S1's 4x internal output is already at
    high resolution. Instead of downscaling to 1x then re-upscaling via S2,
    we take S1's 4x output and resize directly to the target scale.
    This saves one full 4x model pass (~50% faster).
    """
    import cv2
    h, w = frame_bgr.shape[:2]
    target_w = int(w * outscale)
    target_h = int(h * outscale)
    target_w = target_w - (target_w % 2)
    target_h = target_h - (target_h % 2)

    s1.scale = 4
    if hasattr(s1, "dni_weight") and s1.dni_weight is not None:
        s1.dni_weight = [1 - dn, dn]
    # Get the 4x output directly (outscale=4 keeps full model output)
    cleaned_4x, _ = s1.enhance(frame_bgr, outscale=4)
    # Resize from 4x to target scale (e.g., 4x → 3x = downscale, 4x → 2x = downscale)
    if cleaned_4x.shape[1] != target_w or cleaned_4x.shape[0] != target_h:
        cleaned_4x = cv2.resize(cleaned_4x, (target_w, target_h),
                                interpolation=cv2.INTER_LANCZOS4)
    return cleaned_4x


def run_two_stage_sr(s1, s2, frame_bgr, outscale, dn=0.5):
    """Combined Stage 1 + Stage 2. Uses fused path when both stages active."""
    need_s1 = s1 is not None and dn > 0
    need_s2 = s2 is not None and outscale > 1.01

    # Fused path: S1's 4x output → resize to target (skip S2 entirely)
    if need_s1 and need_s2 and outscale <= 4.0:
        return run_fused_s1_s2(s1, s2, frame_bgr, outscale, dn=dn)

    # S1 only (denoise, no upscale)
    if need_s1:
        result = run_stage1(s1, frame_bgr, dn=dn)
    else:
        result = frame_bgr

    # S2 only (upscale, no denoise)
    if need_s2:
        result = run_stage2(s2, result, outscale)

    return result


_face_cache_result = None
_face_cache_frame_idx = -1

def run_face(gfpgan, frame_bgr, face_strength=0.5, frame_idx=0):
    """GFPGAN face restoration with faceless frame skip.

    Runs full face detection every 10 frames. If no faces detected,
    skips restoration for the next 9 frames (faces move slowly).
    weight: 0.0 = aggressive, 1.0 = bypass.
    """
    global _face_cache_result, _face_cache_frame_idx

    # Check face detection cache — skip every 10 frames if no faces found
    if frame_idx > 0 and (frame_idx - _face_cache_frame_idx) < 10:
        if _face_cache_result == "no_faces":
            return frame_bgr

    _, _, restored = gfpgan.enhance(
        frame_bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=face_strength,
    )

    # Cache detection result
    _face_cache_frame_idx = frame_idx
    _face_cache_result = "has_faces" if restored is not None else "no_faces"

    return restored if restored is not None else frame_bgr


def _itm_direct(model, rgb_u8, is_full_range, h, w):
    """Run HDRTVDM directly on the full frame. Returns RGB float [0,1] or None on OOM.

    Takes uint8 RGB and normalizes on GPU to avoid CPU float conversion overhead.
    """
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    inp = rgb_u8
    if pad_h or pad_w:
        inp = np.pad(inp, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)
    try:
        t = t.pin_memory().cuda(non_blocking=True)
        # Normalize on GPU — avoids CPU float32 conversion
        t = normalize_frame_cuda(t, is_full_range)
        with torch.no_grad():
            out = model(t).squeeze(0).clamp(0, 1)
        result = out.permute(1, 2, 0).cpu().numpy()
        del t, out
        if pad_h or pad_w:
            result = result[:h, :w, :]
        return result
    except RuntimeError:
        del t
        torch.cuda.empty_cache()  # only on OOM
        return None


def _itm_guide(model, normalized, h, w, guide_size=960):
    """Guide-based ratio transfer: run model at reduced res, apply gain map at full res."""
    import cv2

    # 1. Downscale to guide_size (HDRTVDM is lightweight, use large guide)
    scale = min(guide_size / h, guide_size / w)
    small_h = int(h * scale) // 4 * 4
    small_w = int(w * scale) // 4 * 4
    small_h = max(small_h, 4)
    small_w = max(small_w, 4)

    small_sdr = cv2.resize(normalized, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Pad to multiple of 4
    pad_h = (4 - small_h % 4) % 4
    pad_w = (4 - small_w % 4) % 4
    small_inp = small_sdr
    if pad_h or pad_w:
        small_inp = np.pad(small_inp, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    # 2. Run HDRTVDM on downscaled frame
    t = torch.from_numpy(small_inp).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        small_hdr = model(t.float()).squeeze(0).clamp(0, 1)
    small_hdr = small_hdr.permute(1, 2, 0).cpu().numpy()
    del t

    # Remove padding from model output
    if pad_h or pad_w:
        small_hdr = small_hdr[:small_h, :small_w, :]

    # 3. Compute per-channel gain map: gain = HDR / SDR
    floor = 1e-4
    gain = small_hdr / np.maximum(small_sdr, floor)

    # Clamp gain to prevent extreme amplification in very dark/bright regions
    gain = np.clip(gain, 0.5, 4.0)

    # 4. Upscale gain map to full resolution with bicubic + Gaussian blur
    gain_full = cv2.resize(gain, (w, h), interpolation=cv2.INTER_CUBIC)
    # Gaussian blur smooths interpolation artifacts — kernel proportional to upscale
    blur_k = int(round(1.0 / scale)) | 1  # ensure odd
    blur_k = max(blur_k, 5)
    blur_k = min(blur_k, 31)
    gain_full = cv2.GaussianBlur(gain_full, (blur_k, blur_k), 0)

    # 5. Apply gain to full-resolution SDR frame
    output = normalized * gain_full
    return np.clip(output, 0, 1)


def run_itm(model, frame_bgr, is_full_range=False, tile_size=512, overlap=64):
    """
    HDRTVDM inverse tone mapping.
    Input:  BGR uint8, limited or full range depending on source
    Output: RGB uint16 [0,65535] -- HDR10 ready

    Strategy: try full-frame direct inference first (best quality).
    If OOM, fall back to guide-based ratio transfer (no tiling seams).

    Direct path uses GPU normalization (normalize_frame_cuda) to avoid
    CPU float32 conversion. Fallback path uses CPU normalization since
    it needs numpy for gain map computation.
    """
    rgb = frame_bgr[..., ::-1].copy()                    # BGR → RGB, contiguous
    h, w, c = rgb.shape

    # Try direct full-frame first (best quality, no artifacts)
    # _itm_direct normalizes on GPU via normalize_frame_cuda
    result = _itm_direct(model, rgb, is_full_range, h, w)
    if result is not None:
        return (result * 65535).astype(np.uint16)

    # OOM fallback: guide-based ratio transfer (seamless, slightly less detail)
    # Uses CPU normalization since gain map computation needs numpy
    normalized = normalize_frame(rgb, is_full_range)
    print(f"  ITM: OOM at {w}x{h}, using guide-based ratio transfer")
    result = _itm_guide(model, normalized, h, w, guide_size=960)
    return (result * 65535).astype(np.uint16)


# ═══════════════════════════════════════════════════════════════════════════
#  ASYNC PIPELINE — threaded stage workers for GPU overlap
# ═══════════════════════════════════════════════════════════════════════════

class _PipelineError:
    """Sentinel passed through queues when a stage thread crashes."""
    def __init__(self, stage, exc):
        self.stage = stage
        self.exc = exc


def stage_worker(name, in_queue, out_queue, process_fn):
    """Generic pipeline stage: pull from in_queue, process, push to out_queue.

    Propagates None sentinel for clean shutdown. On exception, pushes
    _PipelineError so downstream stages and the main thread can detect it.
    """
    try:
        while True:
            item = in_queue.get()
            if item is None or isinstance(item, _PipelineError):
                out_queue.put(item)
                break
            result = process_fn(item)
            out_queue.put(result)
    except Exception as e:
        out_queue.put(_PipelineError(name, e))


_itm_stream = None

def _get_itm_stream():
    """Lazy-init a dedicated CUDA stream for ITM to overlap with other stages."""
    global _itm_stream
    if _itm_stream is None:
        _itm_stream = torch.cuda.Stream()
    return _itm_stream


def run_itm_batch(model, frames_bgr, is_full_range=False):
    """Batched HDRTVDM: process N frames at once through TriSegNet.

    Uses a dedicated CUDA stream to overlap transfer and compute.
    Pinned memory + non_blocking transfers overlap CPU→GPU copy with computation.
    normalize_frame_cuda does normalization on GPU to avoid CPU float conversion.

    Input:  list of BGR uint8 numpy arrays (H, W, 3)
    Output: list of RGB uint16 numpy arrays (H, W, 3) [0, 65535]
    """
    stream = _get_itm_stream()

    # Stack uint8 tensors on CPU with pinned memory for async transfer
    tensors = []
    for frame_bgr in frames_bgr:
        rgb = frame_bgr[..., ::-1].copy()  # BGR → RGB, contiguous copy
        t = torch.from_numpy(rgb).permute(2, 0, 1)  # (3, H, W) uint8
        tensors.append(t)

    with torch.cuda.stream(stream):
        # Stack and transfer as uint8 (3x less bandwidth than float32)
        batch_u8 = torch.stack(tensors, dim=0).pin_memory().cuda(non_blocking=True)
        # Normalize on GPU — avoids CPU float32 conversion entirely
        batch = normalize_frame_cuda(batch_u8, is_full_range)
        del batch_u8
        with torch.no_grad():
            out = model(batch).clamp(0, 1)

    stream.synchronize()

    results = []
    for i in range(out.shape[0]):
        frame = out[i].permute(1, 2, 0).cpu().numpy()
        results.append((frame * 65535).astype(np.uint16))
    del batch, out
    return results


class TemporalSmoother:
    """EMA-based temporal brightness stabilizer for HDR tone-mapped frames.

    Per-frame ITM can produce different brightness levels for adjacent frames.
    This smooths the luminance using an exponential moving average so consecutive
    frames don't flicker.

    alpha: smoothing factor (0 = off, 0.1-0.2 = recommended, 1.0 = no smoothing)
    max_correction: clamp per-frame brightness adjustment to avoid flattening scene cuts
    """

    def __init__(self, alpha=0.15, max_correction=0.15):
        self.alpha = alpha
        self.max_correction = max_correction
        self._ema_mean = None

    def smooth(self, frame_u16):
        if self.alpha <= 0 or self.alpha >= 1.0:
            return frame_u16
        # GPU-accelerated temporal smoothing — avoids CPU float64 ops on full frame
        h, w = frame_u16.shape[:2]
        ch, cw = h // 4, w // 4
        # Sample center crop on CPU (small region, fast)
        sample = frame_u16[ch:h-ch, cw:w-cw]
        cur_mean = float(sample.astype(np.float64).mean())
        if self._ema_mean is None:
            self._ema_mean = cur_mean
            return frame_u16
        self._ema_mean = self.alpha * cur_mean + (1 - self.alpha) * self._ema_mean
        if cur_mean > 0:
            scale = self._ema_mean / cur_mean
            lo = 1.0 - self.max_correction
            hi = 1.0 + self.max_correction
            scale = max(lo, min(hi, scale))
            if abs(scale - 1.0) < 1e-6:
                return frame_u16
            # Scale full frame on GPU — much faster than CPU float64 for 4K frames
            t = torch.from_numpy(frame_u16.astype(np.int32)).cuda().float()
            t = (t * scale).clamp_(0, 65535)
            frame_u16 = t.cpu().to(torch.int32).numpy().astype(np.uint16)
        return frame_u16


def stage4_batch_worker(in_queue, out_queue, model, is_full_range, batch_size=4,
                        temporal_smooth=0.15):
    """Stage 4 worker that collects frames into batches for efficient ITM.

    temporal_smooth: EMA alpha for temporal brightness smoothing (0 = off, 0.1-0.2 recommended).
    """
    batch = []
    smoother = TemporalSmoother(alpha=temporal_smooth)

    try:
        while True:
            item = in_queue.get()
            if item is None or isinstance(item, _PipelineError):
                # Flush remaining partial batch
                if batch:
                    results = run_itm_batch(model, batch, is_full_range)
                    for r in results:
                        out_queue.put(smoother.smooth(r))
                out_queue.put(item)
                break
            batch.append(item)
            if len(batch) >= batch_size:
                results = run_itm_batch(model, batch, is_full_range)
                for r in results:
                    out_queue.put(smoother.smooth(r))
                batch = []
    except Exception as e:
        out_queue.put(_PipelineError("S4", e))


def encoder_writer_sdr(in_queue, encoder_proc, counter, lock):
    """Drain output queue and write to SDR encoder pipe."""
    try:
        while True:
            item = in_queue.get()
            if item is None or isinstance(item, _PipelineError):
                break
            encoder_proc.stdin.write(item.tobytes())
            with lock:
                counter[0] += 1
    except Exception:
        pass
    finally:
        try:
            encoder_proc.stdin.close()
        except Exception:
            pass


def encoder_writer_hdr(in_queue, raw_file, hdr_collector, counter, lock):
    """Drain output queue and write HDR frames to temp file + collect metadata."""
    try:
        while True:
            item = in_queue.get()
            if item is None or isinstance(item, _PipelineError):
                break
            hdr_collector.add_frame(item)
            raw_file.write(item.tobytes())
            with lock:
                counter[0] += 1
    except Exception:
        pass


def run_pipeline_threaded(frame_q, s1, s2, gfpgan, itm_model,
                          scale_factor, args, is_full_range,
                          encoder=None, raw_file=None, hdr_collector=None,
                          total_frames=None):
    """Launch the async threaded pipeline and wait for completion.

    For SDR: pass encoder (subprocess with stdin pipe).
    For HDR: pass raw_file (open file handle) and hdr_collector.
    """
    dn = args.dn
    face_strength = args.face_strength
    temporal_smooth = getattr(args, 'temporal_smooth', 0.15)

    # Dynamic ITM batch size based on VRAM and output resolution
    batch_size = args.batch if args.batch > 0 else 4
    if args.batch <= 0:
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            # ITM processes at output resolution; estimate bytes per frame:
            # ~12 bytes/pixel (FP32 in + out + intermediate) per batch item
            out_pixels = (args.target or 2160) * ((args.target or 2160) * 16 // 9)
            mb_per_frame = out_pixels * 12 / (1024**2)
            # Use ~75% of VRAM for batching — S1/S2 models are unloaded by this
            # point in the pipeline, so more VRAM is available for ITM batches
            available_mb = vram_gb * 1024 * 0.75
            batch_size = max(1, min(32, int(available_mb / mb_per_frame)))
        except Exception:
            batch_size = 4

    # ── Build queue chain dynamically ────────────────────────────────
    q_after_s2 = queue.Queue(maxsize=2)

    if gfpgan is not None:
        q_after_s3 = queue.Queue(maxsize=4)
    else:
        q_after_s3 = q_after_s2  # skip face stage

    if itm_model is not None:
        q_after_s4 = queue.Queue(maxsize=2)
    else:
        q_after_s4 = q_after_s3  # skip ITM stage

    # ── Shared progress counter ──────────────────────────────────────
    counter = [0]
    lock = threading.Lock()

    # ── Launch stage threads ─────────────────────────────────────────
    threads = []

    # Fused Stage 1+2: denoise + upscale in single GPU pass (no CPU round-trip)
    t12 = threading.Thread(
        target=stage_worker, daemon=True,
        args=("S1+S2", frame_q, q_after_s2,
              lambda f: run_two_stage_sr(s1, s2, f, outscale=scale_factor, dn=dn)))
    threads.append(t12)

    # Stage 3: face restoration (optional, with faceless frame skip)
    if gfpgan is not None:
        _face_idx = [0]
        def _face_fn(f):
            idx = _face_idx[0]
            _face_idx[0] += 1
            return run_face(gfpgan, f, face_strength=face_strength, frame_idx=idx)
        t3 = threading.Thread(
            target=stage_worker, daemon=True,
            args=("S3", q_after_s2, q_after_s3, _face_fn))
        threads.append(t3)

    # Stage 4: ITM/HDR (optional, batched)
    if itm_model is not None:
        t4 = threading.Thread(
            target=stage4_batch_worker, daemon=True,
            args=(q_after_s3, q_after_s4, itm_model,
                  is_full_range, batch_size, temporal_smooth))
        threads.append(t4)

    # Encoder/writer thread
    if encoder is not None:
        tw = threading.Thread(
            target=encoder_writer_sdr, daemon=True,
            args=(q_after_s4, encoder, counter, lock))
    else:
        tw = threading.Thread(
            target=encoder_writer_hdr, daemon=True,
            args=(q_after_s4, raw_file, hdr_collector, counter, lock))
    threads.append(tw)

    for t in threads:
        t.start()

    # ── Main thread: progress bar ────────────────────────────────────
    pbar = tqdm(total=total_frames, desc="  Pipeline", unit="frame")
    last_count = 0

    while tw.is_alive():
        tw.join(timeout=0.5)
        with lock:
            current = counter[0]
        delta = current - last_count
        if delta > 0:
            pbar.update(delta)
            last_count = current

    # Final update
    with lock:
        current = counter[0]
    if current > last_count:
        pbar.update(current - last_count)
    pbar.close()

    # Wait for all threads
    for t in threads:
        t.join(timeout=5)

    return counter[0]


# ═══════════════════════════════════════════════════════════════════════════
#  FFMPEG I/O
# ═══════════════════════════════════════════════════════════════════════════

def decode_frames(path, deinterlace=False, is_full_range=False, threads=8,
                   start_time=None, duration=None):
    """
    FFmpeg -> raw BGR24 frame pipe.

    Uses NVDEC hardware acceleration when available — offloads entropy decoding
    and motion compensation to the GPU's dedicated decode chip, reducing CPU load
    on EPYC/Xeon nodes. Falls back to CPU decode transparently.

    For full-range sources, signal -color_range pc to FFmpeg explicitly so it
    doesn't misinterpret levels during pix_fmt conversion to bgr24.
    """
    vf_parts = []
    if deinterlace:
        vf_parts.append("yadif=mode=0")
    vf_parts.append("format=bgr24")

    cmd = ["ffmpeg"]
    if start_time is not None:
        cmd += ["-ss", str(start_time)]
    if _NVDEC_AVAILABLE:
        cmd += ["-hwaccel", "cuda"]
        print("  [DECODE] Using NVDEC hardware decode")
    else:
        print("  [DECODE] Using CPU decode (NVDEC not available)")
    # Explicitly signal input color range — prevents silent level misinterpretation
    cmd += ["-color_range", "pc" if is_full_range else "tv"]
    cmd += ["-threads", str(threads), "-i", str(path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", ",".join(vf_parts), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _has_nvenc():
    """Check if hevc_nvenc is available."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                           capture_output=True, text=True, timeout=5)
        return "hevc_nvenc" in r.stdout
    except Exception:
        return False

_NVENC_AVAILABLE = _has_nvenc()


def _has_nvdec():
    """Check if h264_cuvid decoder is available (NVDEC hardware decode)."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"],
                           capture_output=True, text=True, timeout=5)
        return "h264_cuvid" in r.stdout
    except Exception:
        return False

_NVDEC_AVAILABLE = _has_nvdec()


def encode_sdr(out_w, out_h, fps, output_path, crf=18, preset="p7"):
    """FFmpeg HEVC SDR encode — uses NVENC if available, else libx265."""
    if _NVENC_AVAILABLE:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "-",
            "-c:v", "hevc_nvenc",
            "-preset", preset,
            "-rc", "constqp", "-qp", str(crf),
            "-pix_fmt", "yuv420p",
            output_path,
        ]
    else:
        x265_preset = _map_nvenc_to_x265_preset(preset)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "-",
            "-c:v", "libx265",
            "-preset", x265_preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            output_path,
        ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _build_x265_hdr10_params(max_cll=1000, max_fall=400, extra_params=None):
    """
    Build x265 params string with HDR10 static metadata.

    libx265 is the only FFmpeg encoder that can inject MaxCLL/MaxFALL and
    mastering display SEI messages directly into the HEVC bitstream.
    NVENC (hevc_nvenc) does not support HDR metadata injection.
    """
    # Mastering display: BT.2020 primaries, D65 white point
    # Format: G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)
    # Values in 1/50000 for chromaticity, 1/10000 cd/m2 for luminance
    master_display = (
        "G(13250,34500)"
        "B(7500,3000)"
        "R(34000,16000)"
        "WP(15635,16450)"
        f"L({max(max_cll, 1000) * 10000},50)"
    )

    parts = [
        "hdr-opt=1",
        "repeat-headers=1",
        "colorprim=bt2020",
        "transfer=smpte2084",
        "colormatrix=bt2020nc",
        f"max-cll={max_cll},{max_fall}",
        f"master-display={master_display}",
    ]
    if extra_params:
        parts.append(extra_params)
    return ":".join(parts)


def _map_nvenc_to_x265_preset(preset):
    """Map NVENC preset names (p1-p7) to x265 equivalents."""
    x265_preset_map = {
        "p1": "ultrafast", "p2": "superfast", "p3": "veryfast",
        "p4": "faster", "p5": "medium", "p6": "slow", "p7": "slower",
    }
    return x265_preset_map.get(preset, preset)


def encode_hdr10_static(out_w, out_h, fps, output_path, crf=18, preset="p7",
                        max_cll=1000, max_fall=400):
    """
    FFmpeg libx265 HEVC 10-bit HDR10 encode with static metadata (streaming).

    Uses libx265 to inject MaxCLL/MaxFALL and mastering display SEI messages
    directly into the HEVC bitstream. Accepts raw frames on stdin.
    """
    x265_preset = _map_nvenc_to_x265_preset(preset)
    x265_params = _build_x265_hdr10_params(max_cll=max_cll, max_fall=max_fall)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "rgb48le",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx265",
        "-preset", x265_preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
        "-color_range", "tv",
        "-x265-params", x265_params,
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def encode_hdr10_static_twopass(out_w, out_h, fps, raw_frames_path, output_path,
                                 crf=18, preset="p7", max_cll=1000, max_fall=400):
    """
    Two-pass HDR10 encode: reads raw frames from file, encodes with libx265
    using correct MaxCLL/MaxFALL from the luminance analysis.
    """
    x265_preset = _map_nvenc_to_x265_preset(preset)
    x265_params = _build_x265_hdr10_params(max_cll=max_cll, max_fall=max_fall)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "rgb48le",
        "-r", str(fps), "-i", str(raw_frames_path),
        "-c:v", "libx265",
        "-preset", x265_preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
        "-color_range", "tv",
        "-x265-params", x265_params,
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ENCODE ERROR] FFmpeg exited with code {result.returncode}")
        print(f"  stderr: {result.stderr[-500:] if result.stderr else '(empty)'}")
    return result


def encode_hdr10plus_x265(out_w, out_h, fps, output_path, hdr10plus_json_path,
                          crf=18, preset="slow", max_cll=1000, max_fall=400):
    """
    x265 software encode with HDR10+ dynamic metadata.

    Uses libx265 with --dhdr10-info for per-frame HDR10+ metadata injection.
    Includes static MaxCLL/MaxFALL as fallback for non-HDR10+ displays.
    """
    x265_preset = _map_nvenc_to_x265_preset(preset)
    x265_params = _build_x265_hdr10_params(
        max_cll=max_cll, max_fall=max_fall,
        extra_params=f"dhdr10-info={hdr10plus_json_path}",
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "rgb48le",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx265",
        "-preset", x265_preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
        "-color_range", "tv",
        "-x265-params", x265_params,
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


# ═══════════════════════════════════════════════════════════════════════════
#  CPU/GPU PIPELINE THREADING
# ═══════════════════════════════════════════════════════════════════════════

def build_frame_queue(decode_proc, frame_w, frame_h, maxsize=16):
    """Read raw frames from FFmpeg pipe and queue them for GPU inference.
    Uses readinto + pre-allocated buffer for zero-copy reads.
    Larger buffer (16) keeps GPU fed during decode stalls."""
    q = queue.Queue(maxsize=maxsize)

    def worker():
        frame_bytes = frame_w * frame_h * 3
        buf = bytearray(frame_bytes)
        mv = memoryview(buf)
        while True:
            n = decode_proc.stdout.readinto(mv)
            if not n or n < frame_bytes:
                q.put(None)
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(frame_h, frame_w, 3)
            q.put(frame.copy())  # copy needed — buf is reused next iteration

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return q, t


# ═══════════════════════════════════════════════════════════════════════════
#  15-SECOND INCREMENTAL PREVIEW
# ═══════════════════════════════════════════════════════════════════════════

# Labels shown in logs for each preview stage (cumulative models)
_PREVIEW_STAGE_LABELS = {
    "s1": "realesr-general-x4v3.pth",
    "s2": "realesr-general-x4v3.pth + RealESRGAN_x4plus.pth",
    "s3": "realesr-general-x4v3.pth + RealESRGAN_x4plus.pth + GFPGANv1.4.pth",
    "s4": "realesr-general-x4v3.pth + RealESRGAN_x4plus.pth + GFPGANv1.4.pth + HDRTVDM",
}

# Which stages each preview actually needs to run through
_PREVIEW_STAGE_DEPS = {
    "s1": {"s1"},
    "s2": {"s1", "s2"},
    "s3": {"s1", "s2", "s3"},
    "s4": {"s1", "s2", "s3", "s4"},
}


def run_preview(input_path, output_path, args, info):
    """
    Render a 15-second incremental preview for each selected stage.

    Each preview is cumulative:
      - *_preview_s1.mkv  — realesr-general-x4v3
      - *_preview_s2.mkv  — realesr-general-x4v3 + RealESRGAN_x4plus
      - *_preview_s3.mkv  — + GFPGANv1.4 face restoration
      - *_preview_s4.mkv  — + HDRTVDM HDR tone mapping

    The user selects which stage previews to generate via --preview-stages.
    Only models needed for the highest selected stage are loaded.
    """
    duration = info["duration"]
    preview_duration = min(15.0, duration)
    start_time = max(0, duration * 0.25) if duration > 15 else 0

    w, h = info["width"], info["height"]
    fps = info["fps"]

    # Scale for Stage 2+
    target_h = args.target if args.target > 0 else None
    scale_factor, target_height = get_scale(
        h, target_override=target_h,
        bitrate_kbps=info.get("bitrate_kbps", 0),
        codec=info.get("codec", ""),
        width=w,
    )
    out_w = int(w * scale_factor)
    out_h = target_height
    out_w = out_w + (out_w % 2)
    out_h = out_h + (out_h % 2)

    # Determine which stage previews the user wants
    requested = set(args.preview_stages)

    stem = output_path.stem.replace("_preview_dn", "").replace("_preview", "")
    suffix = output_path.suffix

    # Figure out which processing stages are actually needed
    # (the highest requested preview determines what we run)
    needs = set()
    for s in requested:
        needs |= _PREVIEW_STAGE_DEPS[s]

    need_s1 = "s1" in needs
    need_s2 = "s2" in needs
    need_s3 = "s3" in needs and not args.no_face
    need_s4 = "s4" in needs and not args.no_itm

    # If s3 was requested but face is disabled, skip it silently
    # If s4 was requested but ITM is disabled, skip it silently
    active_previews = []
    for s in ["s1", "s2", "s3", "s4"]:
        if s not in requested:
            continue
        if s == "s3" and args.no_face:
            print(f"  [PREVIEW] Skipping {s} preview (face restoration disabled)")
            continue
        if s == "s4" and args.no_itm:
            print(f"  [PREVIEW] Skipping {s} preview (HDR disabled)")
            continue
        active_previews.append(s)

    if not active_previews:
        print("  [PREVIEW] No stages selected for preview.")
        return {}

    paths = {}
    for s in active_previews:
        paths[s] = output_path.with_name(f"{stem}_preview_{s}{suffix}")

    print(f"\n[PREVIEW] Rendering {preview_duration:.0f}s incremental previews...")
    print(f"  Source   : {w}x{h} @ {fps:.2f} fps")
    print(f"  Start    : {start_time:.1f}s into video")
    if need_s2:
        print(f"  Scale    : {scale_factor:.2f}x -> {out_w}x{out_h}")
    print(f"  Previews :")
    for s in active_previews:
        print(f"    {s}: {_PREVIEW_STAGE_LABELS[s]}")
        print(f"        -> {paths[s].name}")

    # Deinterlace check
    do_deinterlace = False
    if args.deinterlace == "auto":
        do_deinterlace = detect_interlace(input_path)
        if do_deinterlace:
            print("  Deinterlace: detected, applying yadif")
    elif args.deinterlace == "on":
        do_deinterlace = True

    # Color range (force_full_range overrides probe metadata)
    is_full_range = args.force_full_range or info.get("is_full_range", False)

    # Auto-tile sizing based on available VRAM
    tile = args.tile
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            input_pixels = w * h
            # Estimate: 4x model needs ~16 bytes/pixel per tile (FP16)
            # With overhead, safe tile sizes per VRAM:
            #   24GB+ (4090/L40S): tile=0 for ≤720p input, 1024 for ≤1080p, 512 for larger
            #   16GB (A4000/L4):   tile=1024 for ≤720p, 512 for larger
            #   12GB and below:    tile=512 always
            # Auto-detect optimal tile size based on VRAM + input resolution
            # Real-ESRGAN does 4x internal, so input pixels × 16 = output pixels
            # Memory per tile ≈ tile² × 16 bytes (FP16) × model overhead (~10x)
            if vram_gb >= 20:  # 24GB (4090, L40S, A100)
                if input_pixels <= 921600:     # ≤ 720p (1280×720)
                    tile = 0                    # full frame
                elif input_pixels <= 2073600:  # ≤ 1080p
                    tile = 1024
                else:                          # > 1080p
                    tile = 512
            elif vram_gb >= 14:  # 16GB (4080, A4000)
                if input_pixels <= 518400:     # ≤ 480p
                    tile = 0
                elif input_pixels <= 921600:   # ≤ 720p
                    tile = 1024
                else:
                    tile = 512
            else:  # 8-12GB
                tile = 512
            print(f"\n  VRAM: {vram_gb:.1f} GB — tile size: {'full frame' if tile == 0 else tile}")
    except Exception:
        pass

    # Load only the models we need
    print("\n  Loading models...")
    s1_model = s2_model = gfpgan = itm_model = None

    if need_s1:
        # S1 operates at source resolution (outscale=1), so tile size matters
        # less than S2 — but still respect auto-detected tile for large inputs
        s1_tile = 0 if tile == 0 else max(256, tile)
        s1_model = load_stage1(fp16=args.fp16, tile=s1_tile, dn=args.dn)
        print("    Stage 1: realesr-general-x4v3 loaded")

    if need_s2:
        s2_model = load_stage2(fp16=args.fp16, tile=tile, use_trt=not getattr(args, 'no_trt', False))
        print("    Stage 2: RealESRGAN_x4plus loaded")

    if need_s3:
        gfpgan = load_stage3()
        print("    Stage 3: GFPGANv1.4 loaded")

    if need_s4:
        itm_model = load_stage4(args.itm, use_trt=not getattr(args, 'no_trt', False))
        print(f"    Stage 4: HDRTVDM {args.itm} loaded")

    # Decode
    decoder = decode_frames(
        input_path,
        deinterlace=do_deinterlace,
        is_full_range=is_full_range,
        threads=args.workers,
        start_time=start_time,
        duration=preview_duration,
    )
    frame_q, reader_thread = build_frame_queue(decoder, w, h)

    # Open one encoder per selected preview
    encoders = {}
    if "s1" in active_previews:
        encoders["s1"] = encode_sdr(w, h, fps, str(paths["s1"]),
                                    crf=args.crf, preset=args.preset)
    if "s2" in active_previews:
        encoders["s2"] = encode_sdr(out_w, out_h, fps, str(paths["s2"]),
                                    crf=args.crf, preset=args.preset)
    if "s3" in active_previews:
        encoders["s3"] = encode_sdr(out_w, out_h, fps, str(paths["s3"]),
                                    crf=args.crf, preset=args.preset)
    if "s4" in active_previews:
        encoders["s4"] = encode_hdr10_static(
            out_w, out_h, fps, str(paths["s4"]),
            crf=args.crf, preset=args.preset,
            max_cll=1000, max_fall=400,
        )

    # HDR metadata collector for S4 summary
    hdr_collector = HDRMetadataCollector() if need_s4 else None
    itm_smoother = TemporalSmoother(alpha=args.temporal_smooth) if need_s4 else None

    total_frames = int(preview_duration * fps)
    processed = 0
    pbar = tqdm(total=total_frames, desc="  Preview", unit="frame")

    while True:
        frame = frame_q.get()
        if frame is None:
            break

        # Stage 1: artifact cleanup at source resolution
        cleaned = None
        if need_s1:
            cleaned = run_stage1(s1_model, frame, dn=args.dn)
            if "s1" in encoders:
                encoders["s1"].stdin.write(cleaned.tobytes())

        # Stage 2: upscale
        upscaled = None
        if need_s2:
            upscaled = run_stage2(s2_model, cleaned, outscale=scale_factor)
            if "s2" in encoders:
                encoders["s2"].stdin.write(upscaled.tobytes())

        # Stage 3: face restoration
        stage3_out = upscaled
        if need_s3:
            face_result = run_face(gfpgan, upscaled, face_strength=args.face_strength)
            if "s3" in encoders:
                encoders["s3"].stdin.write(face_result.tobytes())
            stage3_out = face_result

        # Stage 4: HDR tone mapping
        if need_s4:
            hdr_frame = run_itm(itm_model, stage3_out, is_full_range=is_full_range)
            hdr_frame = itm_smoother.smooth(hdr_frame)
            hdr_collector.add_frame(hdr_frame)
            if "s4" in encoders:
                encoders["s4"].stdin.write(hdr_frame.tobytes())

        processed += 1
        pbar.update(1)

    pbar.close()

    # Close all encoders
    for enc in encoders.values():
        enc.stdin.close()
    for enc in encoders.values():
        enc.wait()
    decoder.wait()

    # Print HDR metadata summary if applicable
    if hdr_collector is not None and hdr_collector.frame_count > 0:
        hdr_collector.print_summary()

    print(f"\n[PREVIEW] Done! {processed} frames x {len(encoders)} preview(s):")
    for s, p in paths.items():
        print(f"  {s}: {p}")
    return paths


# ═══════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(input_path, output_path, args):
    """Run the full 4-stage pipeline on a single video."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    # ── Probe ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Processing: {input_path.name}")
    print(f"{'='*60}")

    info = probe_video(input_path)
    w, h = info["width"], info["height"]
    fps = info["fps"]
    bitrate = info["bitrate_kbps"]
    duration = info["duration"]

    print(f"\n  Resolution : {w}x{h}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Codec      : {info['codec']}")
    print(f"  Bitrate    : {bitrate} kbps" if bitrate else "  Bitrate    : unknown")
    print(f"  Duration   : {duration:.1f}s")

    # ── Color range validation ─────────────────────────────────────────
    try:
        is_full_range = validate_color_range(info, force_full_range=args.force_full_range)
    except ValueError as e:
        print(f"\n  {_ERR}Aborting: {e}{_RESET}\n")
        return

    # Heuristic pixel check for unsignaled range
    if info.get("color_range") is None and not args.force_full_range:
        check_first_frame_range(input_path)

    # HDR source detection — auto-disable ITM if source is already HDR
    if info.get("is_hdr") and not args.no_itm:
        print(f"  [HDR] Source is already HDR — auto-disabling Stage 4 (ITM)")
        args.no_itm = True

    # ── Bitrate-based DN recommendation ────────────────────────────────
    rec_dn, rec_desc = recommend_dn(h, bitrate)
    print(f"\n  [AUTO] Recommended DN: {rec_dn} -- {rec_desc}")

    if args.dn == -1:
        args.dn = rec_dn
        print(f"  [AUTO] Using recommended DN: {args.dn}")
    else:
        print(f"  [USER] Using specified DN: {args.dn}")

    # ── Face auto-detection ────────────────────────────────────────────
    if not args.no_face:
        print("\n  Scanning for faces...")
        has_faces, face_count, face_desc = detect_faces_in_video(
            input_path, sample_count=5, duration=duration,
            width=w, height=h,
        )
        print(f"  [FACE] {face_desc}")
        if not has_faces:
            print("  [FACE] Auto-skipping Stage 3 (no faces detected)")
            args.no_face = True

    # ── Grain analysis (before processing strips it) ─────────────────
    grain_profile = None
    if getattr(args, "preserve_grain", False):
        print("\n  Analyzing film grain from source...")
        # Read a frame from the middle of the video for representative grain
        mid_time = duration / 2
        cap = cv2.VideoCapture(str(input_path))
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
        ret, sample_frame = cap.read()
        cap.release()
        if ret and sample_frame is not None:
            grain_profile = analyze_grain(sample_frame)
            grain_str = getattr(args, "grain_strength", 1.0)
            print(f"  [GRAIN] Intensity: B={grain_profile['intensity'][0]:.2f} "
                  f"G={grain_profile['intensity'][1]:.2f} "
                  f"R={grain_profile['intensity'][2]:.2f}")
            print(f"  [GRAIN] Coarseness: {grain_profile['coarseness']:.2f}")
            print(f"  [GRAIN] Re-synthesis strength: {grain_str:.1f}x")
        else:
            print("  [GRAIN] Could not read sample frame — grain preservation disabled")

    # ── Deinterlace ────────────────────────────────────────────────────
    do_deinterlace = False
    if args.deinterlace == "auto":
        do_deinterlace = detect_interlace(input_path)
        if do_deinterlace:
            print("\n  [DEINTERLACE] Interlaced source detected, applying yadif")
    elif args.deinterlace == "on":
        do_deinterlace = True
        print("\n  [DEINTERLACE] Forced on")

    # ── Scale logic ────────────────────────────────────────────────────
    target_h = args.target if args.target > 0 else None
    scale_factor, target_height = get_scale(
        h, target_override=target_h,
        bitrate_kbps=bitrate,
        codec=info.get("codec", ""),
        width=w,
    )
    out_w = int(w * scale_factor)
    out_h = target_height
    # Ensure even dimensions (required by encoders)
    out_w = out_w + (out_w % 2)
    out_h = out_h + (out_h % 2)

    print(f"\n  Scale      : {scale_factor:.2f}x ({w}x{h} -> {out_w}x{out_h})")

    # ── HDR mode ───────────────────────────────────────────────────────
    hdr_mode = args.hdr_mode if not args.no_itm else "sdr"
    if hdr_mode == "hdr10plus":
        print(f"  HDR mode   : HDR10+ (dynamic metadata, libx265 software encode)")
    elif hdr_mode == "hdr10":
        print(f"  HDR mode   : HDR10 (static MaxCLL/MaxFALL, libx265 encode)")
    else:
        print(f"  HDR mode   : SDR (no tone mapping)")

    # ── Preview mode ───────────────────────────────────────────────────
    if args.preview:
        preview_path = output_path.with_name(
            output_path.stem + "_preview" + output_path.suffix
        )
        run_preview(input_path, preview_path, args, info)
        return

    # ── Load models ────────────────────────────────────────────────────
    print("\n  Loading models...")
    t0 = time.time()

    # Skip S1 if DN is explicitly 0 (no denoising)
    s1 = None
    if args.dn != 0:
        s1 = load_stage1(fp16=args.fp16, tile=0, dn=args.dn)
        print("    Stage 1: realesr-general-x4v3 loaded")
    else:
        print("    Stage 1: SKIPPED (DN=0, no denoising)")

    # Auto-tile sizing based on VRAM (same logic as preview path)
    tile = args.tile
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            input_pixels = w * h
            if tile > 0:
                if vram_gb >= 20:
                    if input_pixels <= 1280 * 720:
                        tile = 0
                    elif input_pixels <= 1920 * 1080:
                        tile = max(tile, 1024)
                elif vram_gb >= 14:
                    if input_pixels <= 1280 * 720:
                        tile = max(tile, 1024)
            print(f"  VRAM: {vram_gb:.1f} GB — tile size: {'full frame' if tile == 0 else tile}")
    except Exception:
        pass

    # Skip S2 if scale ≈ 1.0 (already at target resolution)
    s2 = None
    if scale_factor > 1.01:
        s2 = load_stage2(fp16=args.fp16, tile=tile, use_trt=not getattr(args, 'no_trt', False))
        print("    Stage 2: RealESRGAN_x4plus loaded")
    else:
        print(f"    Stage 2: SKIPPED (scale={scale_factor:.2f}x, no upscale needed)")

    gfpgan = None
    if not args.no_face:
        gfpgan = load_stage3()
        print("    Stage 3: GFPGANv1.4 loaded")

    itm_model = None
    if not args.no_itm:
        itm_model = load_stage4(args.itm, use_trt=not getattr(args, 'no_trt', False))
        print(f"    Stage 4: HDRTVDM {args.itm} loaded")

    print(f"    All models loaded in {time.time() - t0:.1f}s")

    # ── Decode ─────────────────────────────────────────────────────────
    chunk_start = getattr(args, 'start_time', None)
    chunk_duration = getattr(args, 'duration', None)
    decoder = decode_frames(
        input_path, deinterlace=do_deinterlace,
        is_full_range=is_full_range, threads=args.workers,
        start_time=chunk_start, duration=chunk_duration,
    )
    frame_q, reader_thread = build_frame_queue(decoder, w, h)

    # Override total frames for chunk mode
    if chunk_duration and fps > 0:
        total_frames = int(chunk_duration * fps)
    elif chunk_start is not None and chunk_duration is None and duration > 0:
        total_frames = int((duration - chunk_start) * fps)

    # ── HDR metadata collector ─────────────────────────────────────────
    hdr_collector = HDRMetadataCollector() if not args.no_itm else None

    # ── Encode strategy ────────────────────────────────────────────────
    #
    # HDR10 static: Two-pass approach
    #   Pass 1: process all frames, write raw HDR to temp file, collect metadata
    #   Pass 2: encode from temp file with correct MaxCLL/MaxFALL
    #
    # HDR10+ dynamic: Single-pass approach
    #   Process all frames, write raw HDR to temp file, collect per-frame metadata,
    #   then write HDR10+ JSON and encode with x265 --dhdr10-info
    #
    # SDR: Single-pass direct pipe
    #
    # Both HDR modes need frame analysis before encode, so we buffer to disk.
    # SDR mode pipes directly.
    # ───────────────────────────────────────────────────────────────────

    # Only set total_frames from full duration if not already set by chunk mode
    if chunk_duration is None and chunk_start is None:
        total_frames = int(duration * fps) if duration > 0 else None
    stages = []
    if s1 is not None:
        stages.append("S1")
    if s2 is not None:
        stages.append("S2")
    if not args.no_face:
        stages.append("S3")
    if not args.no_itm:
        stages.append("S4")
    if grain_profile is not None:
        stages.append("GRAIN")
    stages_desc = "+".join(stages) if stages else "PASSTHROUGH"

    if hdr_mode == "sdr":
        # ── SDR: direct pipe to encoder ───────────────────────────────
        print(f"\n  Processing [{stages_desc}] -> SDR output...")
        pbar = tqdm(total=total_frames, desc="  Pipeline", unit="frame")
        processed = 0
        encoder = None

        while True:
            frame = frame_q.get()
            if frame is None:
                break
            result = run_two_stage_sr(s1, s2, frame, outscale=scale_factor, dn=args.dn)
            if gfpgan is not None:
                result = run_face(gfpgan, result, face_strength=args.face_strength)
            # Re-apply grain if preserve_grain is enabled
            if grain_profile is not None:
                result = synthesize_grain(result, grain_profile,
                                          strength=getattr(args, "grain_strength", 1.0))
            # Create encoder on first frame with actual dimensions (must be even for x265)
            if encoder is None:
                out_h, out_w = result.shape[:2]
                out_w = out_w - (out_w % 2)
                out_h = out_h - (out_h % 2)
                if result.shape[1] != out_w or result.shape[0] != out_h:
                    result = result[:out_h, :out_w]
                print(f"  Actual output: {out_w}x{out_h}")
                encoder = encode_sdr(
                    out_w, out_h, fps, str(output_path),
                    crf=args.crf, preset=args.preset,
                )
            # Ensure even dimensions on every frame
            if result.shape[1] != out_w or result.shape[0] != out_h:
                result = result[:out_h, :out_w]
            encoder.stdin.write(result.tobytes())
            processed += 1
            pbar.update(1)

        pbar.close()
        if encoder is not None:
            encoder.stdin.close()
            encoder.wait()
        decoder.wait()
        print(f"\n  Done! {processed} frames -> {output_path}")

    else:
        # ── HDR10 or HDR10+: buffer to temp, analyze, then encode ──────
        temp_dir = output_path.parent
        raw_temp = temp_dir / f".{output_path.stem}_raw_hdr.rgb48"

        print(f"\n  Processing [{stages_desc}] -> buffering HDR frames...")
        pbar = tqdm(total=total_frames, desc="  Pipeline", unit="frame")
        processed = 0
        itm_smoother = TemporalSmoother(alpha=args.temporal_smooth)

        try:
            with open(raw_temp, "wb") as raw_f:
                while True:
                    frame = frame_q.get()
                    if frame is None:
                        break
                    result = run_two_stage_sr(
                        s1, s2, frame, outscale=scale_factor, dn=args.dn,
                    )
                    if gfpgan is not None:
                        result = run_face(
                            gfpgan, result, face_strength=args.face_strength,
                        )
                    hdr_frame = run_itm(itm_model, result, is_full_range=is_full_range)
                    hdr_frame = itm_smoother.smooth(hdr_frame)
                    # Re-apply grain if preserve_grain is enabled (16-bit HDR)
                    if grain_profile is not None:
                        hdr_frame = synthesize_grain(hdr_frame, grain_profile,
                                                      strength=getattr(args, "grain_strength", 1.0))
                    # Update actual output dimensions from first frame (must be even for x265)
                    if processed == 0:
                        out_h, out_w = hdr_frame.shape[:2]
                        out_w = out_w - (out_w % 2)
                        out_h = out_h - (out_h % 2)
                        if hdr_frame.shape[1] != out_w or hdr_frame.shape[0] != out_h:
                            hdr_frame = hdr_frame[:out_h, :out_w]
                        print(f"  Actual output: {out_w}x{out_h}")
                    # Ensure even dimensions on every frame
                    if hdr_frame.shape[1] != out_w or hdr_frame.shape[0] != out_h:
                        hdr_frame = hdr_frame[:out_h, :out_w]
                    hdr_collector.add_frame(hdr_frame)
                    raw_f.write(hdr_frame.tobytes())
                    processed += 1
                    pbar.update(1)

            pbar.close()
            decoder.wait()

            # Print metadata summary
            hdr_collector.print_summary()

            # Verify raw frames were written
            raw_size = raw_temp.stat().st_size if raw_temp.exists() else 0
            expected_size = processed * out_w * out_h * 6  # rgb48le = 6 bytes/pixel
            print(f"  Raw buffer: {raw_size / 1048576:.1f} MB "
                  f"(expected {expected_size / 1048576:.1f} MB for {processed} frames @ {out_w}x{out_h})")
            if raw_size == 0:
                print(f"  [ERROR] Raw frame buffer is empty — no frames were written!")

            if hdr_mode == "hdr10":
                print(f"\n  Encoding HDR10 with MaxCLL={hdr_collector.max_cll}, "
                      f"MaxFALL={hdr_collector.max_fall}...")

                encode_hdr10_static_twopass(
                    out_w, out_h, fps,
                    raw_frames_path=str(raw_temp),
                    output_path=str(output_path),
                    crf=args.crf,
                    preset=args.preset,
                    max_cll=hdr_collector.max_cll,
                    max_fall=hdr_collector.max_fall,
                )
                print(f"  Done! {processed} frames -> {output_path}")

            elif hdr_mode == "hdr10plus":
                json_path = temp_dir / f".{output_path.stem}_hdr10plus.json"
                metadata = hdr_collector.generate_hdr10plus_json()

                with open(json_path, "w") as jf:
                    json.dump(metadata, jf)
                print(f"  HDR10+ metadata: {len(metadata['SceneInfo'])} frames -> {json_path.name}")

                print(f"\n  Encoding HDR10+ with x265 (software)...")
                print(f"    MaxCLL={hdr_collector.max_cll}, MaxFALL={hdr_collector.max_fall}")
                print(f"    Per-frame dynamic metadata: {json_path.name}")

                encoder = encode_hdr10plus_x265(
                    out_w, out_h, fps, str(output_path),
                    hdr10plus_json_path=str(json_path),
                    crf=args.crf,
                    preset=args.preset,
                    max_cll=hdr_collector.max_cll,
                    max_fall=hdr_collector.max_fall,
                )

                frame_size = out_w * out_h * 6
                with open(raw_temp, "rb") as rf:
                    while True:
                        chunk = rf.read(frame_size)
                        if not chunk or len(chunk) < frame_size:
                            break
                        encoder.stdin.write(chunk)

                encoder.stdin.close()
                encoder.wait()

                try:
                    json_path.unlink()
                except OSError:
                    pass

                print(f"  Done! {processed} frames -> {output_path}")

        finally:
            try:
                raw_temp.unlink()
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="AI Upscale & HDR Pipeline -- 4-stage video processing",
    )
    p.add_argument("-i", "--input", required=True, help="Input video or directory")
    p.add_argument("-o", "--output", required=True, help="Output file or directory")
    p.add_argument(
        "--dn", type=float, default=-1,
        help="Stage 1 denoising strength (0.0-1.0). -1 = auto-recommend from bitrate.",
    )
    p.add_argument(
        "--face-strength", type=float, default=0.5,
        help="GFPGAN blend weight (0.0 = aggressive, 1.0 = bypass). Default: 0.5",
    )
    p.add_argument("--itm", default="params_3DM", help="HDRTVDM checkpoint name")
    p.add_argument("--fp16", action="store_true", default=True, help="FP16 inference")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--batch", type=int, default=4,
                   help="ITM batch size (frames per GPU batch for Stage 4)")
    p.add_argument("--tile", type=int, default=0, help="ESRGAN tile size (0=disabled)")
    p.add_argument(
        "--deinterlace", choices=["auto", "on", "off"], default="auto",
        help="Deinterlace mode",
    )
    p.add_argument(
        "--target", type=int, default=0,
        help="Force output height (e.g. 2160). 0 = adaptive.",
    )
    p.add_argument("--crf", type=int, default=18, help="Encoder quality (0-51, lower=better)")
    p.add_argument("--preset", default="p7", help="Encoder preset (p1-p7, maps to x265 for HDR)")
    p.add_argument("--no-face", action="store_true", help="Skip Stage 3 (face restore)")
    p.add_argument("--no-itm", action="store_true", help="Skip Stage 4 (HDR), output SDR")
    p.add_argument("--no-trt", action="store_true", help="Disable TensorRT acceleration")
    p.add_argument(
        "--preserve-grain", action="store_true",
        help=(
            "Analyze film grain from source, then re-apply matching synthetic grain "
            "to the output. Useful for music videos and stylistic content."
        ),
    )
    p.add_argument(
        "--grain-strength", type=float, default=1.0,
        help="Grain re-synthesis strength (0.0-2.0). 1.0 = match original. Default: 1.0",
    )
    p.add_argument(
        "--temporal-smooth", type=float, default=0.15,
        help=(
            "HDR temporal brightness smoothing (EMA alpha). "
            "0 = off, 0.15 = default, higher = more responsive. "
            "Prevents per-frame brightness flickering from ITM."
        ),
    )
    p.add_argument("--workers", type=int, default=8, help="FFmpeg decode threads")
    p.add_argument(
        "--force-full-range", action="store_true",
        help=(
            "Force treat input as full range (pc). Use when metadata is missing "
            "but source is known full range (screen recordings, GoPro, iPhone, etc.)."
        ),
    )
    p.add_argument(
        "--preview", action="store_true",
        help="Render 15s incremental previews for selected stages.",
    )
    p.add_argument(
        "--preview-stages", nargs="+", default=["s1", "s2", "s3", "s4"],
        choices=["s1", "s2", "s3", "s4"],
        help=(
            "Which stage previews to generate (default: all). Each is cumulative: "
            "s1 = realesr-general-x4v3, "
            "s2 = s1 + RealESRGAN_x4plus, "
            "s3 = s2 + GFPGANv1.4, "
            "s4 = s3 + HDRTVDM."
        ),
    )
    p.add_argument(
        "--hdr-mode", choices=["hdr10", "hdr10plus"], default="hdr10",
        help=(
            "HDR output mode. "
            "hdr10 = static MaxCLL/MaxFALL via libx265. "
            "hdr10plus = per-frame dynamic metadata via libx265 (slower, better). "
            "Default: hdr10."
        ),
    )
    # Chunk processing (for multi-GPU splitting)
    p.add_argument("--start-time", type=float, default=None,
                   help="Start time in seconds (for chunk processing)")
    p.add_argument("--duration", type=float, default=None,
                   help="Duration in seconds (for chunk processing)")
    p.add_argument("--raw-output", action="store_true",
                   help="Output raw RGB48 frames + metadata JSON instead of encoded video (for chunk stitching)")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Batch mode: input is a directory
    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".ts", ".m2ts", ".wmv", ".flv"}
        videos = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in video_exts
        )
        if not videos:
            print(f"No video files found in {input_path}")
            sys.exit(1)

        print(f"\nBatch mode: {len(videos)} video(s) in {input_path}")
        for video in videos:
            # Sanitize filename — remove chars that break FFmpeg subprocess calls
            safe_stem = re.sub(r'[|&;<>\"\'`$\\!]', '_', video.stem)
            out_name = safe_stem + ("_preview" if args.preview else "_hdr") + ".mkv"
            out = output_path / out_name
            run_pipeline(str(video), str(out), args)
    else:
        # Single file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_pipeline(str(input_path), str(output_path), args)

    print("\nAll done!")


if __name__ == "__main__":
    main()
