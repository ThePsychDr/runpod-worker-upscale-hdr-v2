"""
grade_detect.py — Color Grade Detection Module
AI Video Upscaling + HDR10 Pipeline

Analyzes sampled frames from a video to detect the intensity and style of
color grading, then recommends appropriate --itm-chroma-protect settings
to preserve the grade during HDRTVDM inverse tone mapping.

Usage:
    from grade_detect import analyze_color_grade, resolve_itm_mode

    grade = analyze_color_grade(input_path, probe_info)
    itm_mode, protect_val = resolve_itm_mode(args, probe_info, input_path)

Standalone CLI:
    python grade_detect.py input.mp4
"""

import subprocess
import json
import argparse
import numpy as np
import cv2

# ─── ANSI colors for terminal output ─────────────────────────────────────────

_WARN  = '\033[93m'
_ERR   = '\033[91m'
_OK    = '\033[92m'
_INFO  = '\033[94m'
_BOLD  = '\033[1m'
_RESET = '\033[0m'


# ─── Grade thresholds ─────────────────────────────────────────────────────────
# Tune these after running on your own library.
# Higher sensitivity = more content treated as graded.

SENSITIVITY_PRESETS = {
    'low': {
        # Only flags obvious, heavy grades
        'bw_std_threshold':        3.0,
        'desat_range_threshold':   25,
        'crushed_luma_floor':      35,
        'bias_threshold':          18,
        'intensity_protect_low':   0.5,
        'intensity_protect_high':  0.75,
        'intensity_luma_only':     0.80,
    },
    'medium': {
        # Default — catches most music video grades
        'bw_std_threshold':        4.0,
        'desat_range_threshold':   35,
        'crushed_luma_floor':      30,
        'bias_threshold':          12,
        'intensity_protect_low':   0.2,
        'intensity_protect_high':  0.4,
        'intensity_luma_only':     0.65,
    },
    'high': {
        # Treats subtle grades as graded — more protection, less HDRTVDM chroma
        'bw_std_threshold':        6.0,
        'desat_range_threshold':   50,
        'crushed_luma_floor':      25,
        'bias_threshold':          8,
        'intensity_protect_low':   0.1,
        'intensity_protect_high':  0.25,
        'intensity_luma_only':     0.50,
    },
}


def _grade_unknown():
    return {
        'grade_intensity':     None,
        'dominant_style':      'unknown',
        'is_bw':               False,
        'recommended_protect': 0.0,
        'use_luma_only':       False,
        'signals':             {},
        'auto_detect':         False,
    }


def analyze_color_grade(path, probe_info, num_samples=12, skip_pct=0.10,
                         sensitivity='medium'):
    """
    Samples frames spread across the middle of the video and measures chroma
    statistics to estimate how intensely the content is color graded.

    Skips the first and last skip_pct of duration to avoid:
      - Intro logos, title cards, fade-from-black openings
      - Credits, end cards, fade-to-black closings
      - Any solid-color bumpers that would skew the statistics

    Args:
        path:        Input video file path
        probe_info:  Dict from probe_video() — needs 'duration', 'width', 'height'
        num_samples: Number of frames to sample (default 12)
        skip_pct:    Fraction to skip at start and end (default 0.10 = 10%)
        sensitivity: 'low' | 'medium' | 'high' — threshold preset

    Returns dict:
        grade_intensity:     float 0.0–1.0 (0=natural, 1=extreme grade)
        dominant_style:      str — 'natural' | 'teal_orange' | 'desaturated' |
                             'crushed_blacks' | 'bleach_bypass' | 'warm' | 'cold' | 'bw'
        is_bw:               bool — True if black and white / near-monochrome
        recommended_protect: float — suggested --itm-chroma-protect value
        use_luma_only:       bool — True if luma-only mode is more appropriate
        signals:             dict — raw measurements for debugging
        auto_detect:         bool — True (indicates this was auto-detected)
    """
    thresh = SENSITIVITY_PRESETS.get(sensitivity, SENSITIVITY_PRESETS['medium'])

    duration = float(probe_info.get('duration', 0))
    width    = probe_info.get('width')
    height   = probe_info.get('height')

    if not duration or not width or not height:
        print(f"  {_WARN}[WARN] Cannot analyze grade — missing duration or dimensions.{_RESET}")
        return _grade_unknown()

    # Build evenly-spaced sample timestamps across the middle of the video
    start_t  = duration * skip_pct
    end_t    = duration * (1.0 - skip_pct)
    usable   = end_t - start_t

    if usable <= 0:
        timestamps = [duration / 2.0]
    else:
        step = usable / (num_samples + 1)
        timestamps = [start_t + step * (i + 1) for i in range(num_samples)]

    print(f"  {_INFO}[GRADE] Analyzing {len(timestamps)} frames "
          f"(skipping first/last {int(skip_pct*100)}% of video, "
          f"sensitivity={sensitivity})...{_RESET}")

    frame_stats = []

    for ts in timestamps:
        cmd = [
            'ffmpeg', '-ss', str(ts), '-i', path,
            '-vframes', '1',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-'
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)

        expected = width * height * 3
        if not result.stdout or len(result.stdout) < expected:
            continue

        frame_bgr = np.frombuffer(result.stdout[:expected], dtype=np.uint8) \
                      .reshape(height, width, 3)

        # Downsample for speed — 1/4 resolution is plenty for histogram stats
        small = cv2.resize(frame_bgr, (width // 4, height // 4))
        frame_ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb).astype(np.float32)

        y  = frame_ycrcb[:, :, 0].flatten()
        cr = frame_ycrcb[:, :, 1].flatten()
        cb = frame_ycrcb[:, :, 2].flatten()

        frame_stats.append({
            # Luma stats
            'y_mean':   float(np.mean(y)),
            'y_std':    float(np.std(y)),
            'y_p05':    float(np.percentile(y, 5)),    # shadow floor
            'y_p95':    float(np.percentile(y, 95)),   # highlight ceiling

            # Chroma spread — natural footage has wide spread (~60–80)
            'cr_std':   float(np.std(cr)),
            'cb_std':   float(np.std(cb)),

            # Chroma range — compressed = desaturated grade
            'cr_range': float(np.percentile(cr, 95) - np.percentile(cr, 5)),
            'cb_range': float(np.percentile(cb, 95) - np.percentile(cb, 5)),

            # Chroma mean — grades shift the mean away from neutral (128)
            'cr_mean':  float(np.mean(cr)),
            'cb_mean':  float(np.mean(cb)),
            'cr_bias':  float(abs(np.mean(cr) - 128)),
            'cb_bias':  float(abs(np.mean(cb) - 128)),
        })

    if not frame_stats:
        print(f"  {_WARN}[WARN] Could not decode any sample frames.{_RESET}")
        return _grade_unknown()

    # ── Aggregate across all sampled frames ───────────────────────────────────

    def avg(key):
        return float(np.mean([f[key] for f in frame_stats]))

    cr_std    = avg('cr_std')
    cb_std    = avg('cb_std')
    cr_range  = avg('cr_range')
    cb_range  = avg('cb_range')
    cr_bias   = avg('cr_bias')
    cb_bias   = avg('cb_bias')
    y_p05     = avg('y_p05')
    y_std     = avg('y_std')
    cr_mean   = avg('cr_mean')
    cb_mean   = avg('cb_mean')

    t = thresh   # shorthand

    # ── Detection logic ───────────────────────────────────────────────────────

    # B&W: both Cr and Cb nearly flat — unambiguous
    is_bw = (cr_std < t['bw_std_threshold'] and cb_std < t['bw_std_threshold'])

    # Desaturated: chroma range compressed below natural baseline
    is_desaturated = (cr_range < t['desat_range_threshold'] or
                      cb_range < t['desat_range_threshold'])

    # Crushed blacks: shadow floor lifted — 5th percentile luma > floor threshold
    # Note: limited-range black is 16, so floor > 16 means intentional crush
    is_crushed = (y_p05 > t['crushed_luma_floor'])

    # Teal/orange: warm skin (Cr > 128) + teal shadows (Cb < 128)
    is_teal_orange = (cr_mean > 128 + t['bias_threshold'] and
                      cb_mean < 128 - t['bias_threshold'] / 2)

    # Warm grade: Cb suppressed uniformly
    is_warm = (cb_mean < 128 - t['bias_threshold'] and not is_teal_orange)

    # Cold/teal: Cb elevated uniformly
    is_cold = (cb_mean > 128 + t['bias_threshold'])

    # Bleach bypass: desaturated + high luma contrast
    is_bleach = (is_desaturated and y_std > 55)

    # ── Grade intensity score ─────────────────────────────────────────────────
    # Each signal contributes 0.0–1.0, final score is weighted average.
    # Natural footage scores near 0.0, heavy grades score 0.6+.

    signals_norm = [
        min(cr_bias / 20.0, 1.0),                       # chroma Cr bias
        min(cb_bias / 20.0, 1.0),                       # chroma Cb bias
        max(0.0, (45 - cr_range) / 45.0),               # Cr compression
        max(0.0, (45 - cb_range) / 45.0),               # Cb compression
        min(max(0.0, y_p05 - 16) / 30.0, 1.0),          # shadow crush
        1.0 if is_bw else 0.0,                           # B&W override
    ]
    grade_intensity = float(np.mean(signals_norm))

    # ── Dominant style ────────────────────────────────────────────────────────

    if is_bw:
        dominant_style = 'bw'
    elif is_bleach:
        dominant_style = 'bleach_bypass'
    elif is_teal_orange:
        dominant_style = 'teal_orange'
    elif is_desaturated:
        dominant_style = 'desaturated'
    elif is_crushed:
        dominant_style = 'crushed_blacks'
    elif is_warm:
        dominant_style = 'warm'
    elif is_cold:
        dominant_style = 'cold'
    else:
        dominant_style = 'natural'

    # ── Recommended ITM settings ──────────────────────────────────────────────

    use_luma_only = is_bw or is_bleach or grade_intensity > t['intensity_luma_only']

    if use_luma_only:
        recommended_protect = 1.0
    elif grade_intensity > t['intensity_protect_high']:
        recommended_protect = 0.8
    elif grade_intensity > t['intensity_protect_low']:
        recommended_protect = 0.5
    else:
        recommended_protect = 0.0   # natural — full HDRTVDM

    result = {
        'grade_intensity':     round(grade_intensity, 3),
        'dominant_style':      dominant_style,
        'is_bw':               is_bw,
        'recommended_protect': recommended_protect,
        'use_luma_only':       use_luma_only,
        'auto_detect':         True,
        'signals': {
            'cr_std':   round(cr_std, 1),
            'cb_std':   round(cb_std, 1),
            'cr_range': round(cr_range, 1),
            'cb_range': round(cb_range, 1),
            'cr_bias':  round(cr_bias, 1),
            'cb_bias':  round(cb_bias, 1),
            'y_p05':    round(y_p05, 1),
            'y_std':    round(y_std, 1),
            'cr_mean':  round(cr_mean, 1),
            'cb_mean':  round(cb_mean, 1),
        },
        'frames_analyzed': len(frame_stats),
    }

    # ── Print summary ─────────────────────────────────────────────────────────

    intensity_bar = '█' * int(grade_intensity * 20) + '░' * (20 - int(grade_intensity * 20))
    color = _ERR if grade_intensity > 0.6 else (_WARN if grade_intensity > 0.3 else _OK)

    print(f"\n  {_BOLD}Grade Analysis Results{_RESET}")
    print(f"  {'─'*44}")
    print(f"  Style:     {_BOLD}{dominant_style}{_RESET}")
    print(f"  Intensity: {color}[{intensity_bar}] {grade_intensity:.2f}{_RESET}")
    print(f"  B&W:       {'yes' if is_bw else 'no'}")
    print(f"  {'─'*44}")
    print(f"  Signals:   Cr bias={result['signals']['cr_bias']}  "
          f"Cb bias={result['signals']['cb_bias']}  "
          f"Cr range={result['signals']['cr_range']}  "
          f"Cb range={result['signals']['cb_range']}")
    print(f"             Y floor={result['signals']['y_p05']}  "
          f"Y std={result['signals']['y_std']}")
    print(f"  {'─'*44}")

    if use_luma_only:
        print(f"  {_BOLD}Recommendation: --itm-luma-only{_RESET}")
        print(f"  Grade too strong for blend — use luma-only expansion")
    elif recommended_protect > 0:
        print(f"  {_BOLD}Recommendation: --itm-chroma-protect {recommended_protect}{_RESET}")
    else:
        print(f"  {_BOLD}Recommendation: full HDRTVDM (natural content){_RESET}")
    print()

    return result


def resolve_itm_mode(args, probe_info, input_path):
    """
    Determines the effective ITM mode for a file.

    Priority:
      1. User-specified flags (--itm-luma-only, --itm-chroma-protect, --no-itm)
      2. Auto-detect via analyze_color_grade()

    Returns:
        (mode, protect_value)
        mode: 'luma_only' | 'protect' | 'full' | 'disabled'
        protect_value: float (chroma_protect value) or None
    """
    # ── User-specified — highest priority ─────────────────────────────────────

    if getattr(args, 'itm_luma_only', False):
        print(f"  {_INFO}[ITM] Mode: luma-only (user specified){_RESET}")
        return 'luma_only', 1.0

    if getattr(args, 'itm_chroma_protect', None) is not None:
        val = args.itm_chroma_protect
        print(f"  {_INFO}[ITM] Mode: chroma-protect {val} (user specified){_RESET}")
        return 'protect', float(val)

    if getattr(args, 'no_itm', False):
        print(f"  {_INFO}[ITM] Mode: disabled (user specified){_RESET}")
        return 'disabled', None

    # ── Auto-detect ───────────────────────────────────────────────────────────

    sensitivity = getattr(args, 'grade_sensitivity', 'medium')
    grade = analyze_color_grade(input_path, probe_info, sensitivity=sensitivity)

    if grade['grade_intensity'] is None:
        print(f"  {_WARN}[ITM] Grade analysis inconclusive — using full HDRTVDM{_RESET}")
        return 'full', 0.0

    if grade['use_luma_only']:
        print(f"  {_INFO}[ITM] Auto: luma-only "
              f"(intensity={grade['grade_intensity']}, style={grade['dominant_style']}){_RESET}")
        return 'luma_only', 1.0

    if grade['recommended_protect'] > 0:
        val = grade['recommended_protect']
        print(f"  {_INFO}[ITM] Auto: chroma-protect {val} "
              f"(intensity={grade['grade_intensity']}, style={grade['dominant_style']}){_RESET}")
        return 'protect', val

    print(f"  {_INFO}[ITM] Auto: full HDRTVDM "
          f"(intensity={grade['grade_intensity']}, natural content){_RESET}")
    return 'full', 0.0


# ─── Standalone CLI ───────────────────────────────────────────────────────────

def _probe_for_cli(path):
    """Minimal probe for standalone CLI use."""
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
           '-show_streams', '-show_format', path]
    data = json.loads(subprocess.check_output(cmd))
    vs = next(s for s in data['streams'] if s['codec_type'] == 'video')
    n, d = map(int, vs['r_frame_rate'].split('/'))
    return {
        'width':    vs['width'],
        'height':   vs['height'],
        'fps':      n / d,
        'duration': float(vs.get('duration') or
                          data.get('format', {}).get('duration', 0)),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze color grade of a video and recommend ITM settings'
    )
    parser.add_argument('input', help='Input video file')
    parser.add_argument('--samples', type=int, default=12,
                        help='Number of frames to sample (default: 12)')
    parser.add_argument('--skip', type=float, default=0.10,
                        help='Fraction to skip at start/end (default: 0.10)')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'],
                        default='medium', help='Detection sensitivity (default: medium)')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    cli_args = parser.parse_args()

    probe = _probe_for_cli(cli_args.input)
    result = analyze_color_grade(
        cli_args.input, probe,
        num_samples=cli_args.samples,
        skip_pct=cli_args.skip,
        sensitivity=cli_args.sensitivity
    )

    if cli_args.json:
        print(json.dumps(result, indent=2))
    else:
        # Recommendation summary
        print(f"{_BOLD}Suggested command flags:{_RESET}")
        if result['use_luma_only']:
            print(f"  --itm-luma-only --face-strength 1.0")
        elif result['recommended_protect'] > 0:
            print(f"  --itm-chroma-protect {result['recommended_protect']} --face-strength 1.0")
        else:
            print(f"  (no changes needed — natural content)")
