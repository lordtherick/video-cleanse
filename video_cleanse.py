#!/usr/bin/env python3
"""
video_cleanse.py - find a word on screen with OCR and blur or block it out.

All settings come from environment variables (see .env), so the same image
can be reused for any video:

  VIDEO_FILENAME   Input video, relative to the /data mount        (required)
  TARGET_WORD      Word(s) to redact, comma-separated, any case     (required)
  START_SECONDS    Start of the section to review                   (default 0)
  END_SECONDS      End of the section to review; blank/0 = to end   (default end)
  REDACT_MODE      "blur" or "block"                                (default blur)
  OUTPUT_FILENAME  Output name; blank = <input>_redacted.mp4        (default blank)

Tuning (optional):
  OCR_INTERVAL     Run OCR every N frames                           (default 5)
  HOLD_FRAMES      Keep a box active N frames after last seen       (default 10)
  PADDING          Extra pixels around each detected box            (default 10)
  MIN_CONFIDENCE   Minimum OCR confidence 0-1                       (default 0.3)
  BLUR_STRENGTH    Blur kernel size (odd number, bigger = heavier)  (default 51)
  BLOCK_COLOUR     Block fill colour as R,G,B                       (default 0,0,0)
  CRF              Quality level (0 = auto-match source bitrate)    (default 0)
  PRESET           Encoder preset                                   (default fast)
  USE_GPU          true to use CUDA if available                    (default false)

Frames outside START_SECONDS..END_SECONDS are passed through untouched in a single
continuous encode matching the source video's codec and bitrate, avoiding any
MP4 codec parameter (SPS/PPS/VPS) splice corruption while keeping the original file size.
"""
import json
import os
import subprocess
import sys
import time
import warnings

# Harmless torch warnings on CPU-only hosts (pin_memory on every OCR call, quantization deprecation at load).
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*quantiz.*")

import cv2
import easyocr
import torch

DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODEL_DIR = os.environ.get("EASYOCR_MODEL_DIR", "/models")


def env(name, default=None):
    value = os.environ.get(name, "")
    return value.strip() if value.strip() != "" else default


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# --- Configuration from environment ---------------------------------------
video_filename = env("VIDEO_FILENAME")
target_raw = env("TARGET_WORD")
if not video_filename:
    fail("VIDEO_FILENAME is not set")
if not target_raw:
    fail("TARGET_WORD is not set")

TARGET_WORDS = [w.strip().lower() for w in target_raw.split(",") if w.strip()]
MIN_SUBSTRING_LEN = max(0, int(env("MIN_SUBSTRING_LEN", 4)))


def build_search_substrings(words, min_len=4):
    """
    Generate all consecutive character substrings of length `min_len` from each target word
    (or the full word itself if `min_len <= 0` or the word is shorter than `min_len`).
    """
    subs = []
    seen = set()
    for w in words:
        if min_len <= 0 or len(w) <= min_len:
            if w not in seen:
                subs.append(w)
                seen.add(w)
        else:
            for i in range(len(w) - min_len + 1):
                sub = w[i:i + min_len]
                if sub not in seen:
                    subs.append(sub)
                    seen.add(sub)
    return subs


SEARCH_PATTERNS = build_search_substrings(TARGET_WORDS, MIN_SUBSTRING_LEN)

try:
    START_SECONDS = float(env("START_SECONDS", 0))
    END_SECONDS = float(env("END_SECONDS", 0))
    OCR_INTERVAL = max(1, int(env("OCR_INTERVAL", 5)))
    HOLD_FRAMES = max(0, int(env("HOLD_FRAMES", 10)))
    PADDING = max(0, int(env("PADDING", 10)))
    MIN_CONFIDENCE = float(env("MIN_CONFIDENCE", 0.3))
    BLUR_STRENGTH = int(env("BLUR_STRENGTH", 51))
    CRF = int(env("CRF", 0))
    PROGRESS_SECONDS = max(1.0, float(env("PROGRESS_SECONDS", 10)))
except ValueError as e:
    fail(f"invalid numeric setting: {e}")

REDACT_MODE = env("REDACT_MODE", "blur").lower()
if REDACT_MODE not in ("blur", "block"):
    fail("REDACT_MODE must be 'blur' or 'block'")

if BLUR_STRENGTH % 2 == 0:
    BLUR_STRENGTH += 1  # Gaussian kernel must be odd

try:
    r, g, b = [int(c) for c in env("BLOCK_COLOUR", "0,0,0").split(",")]
    BLOCK_COLOUR = (b, g, r)  # OpenCV uses BGR
except ValueError:
    fail("BLOCK_COLOUR must be R,G,B e.g. 0,0,0")

PRESET = env("PRESET", "fast")
USE_GPU = env("USE_GPU", "false").lower() in ("1", "true", "yes")

input_path = os.path.join(DATA_DIR, video_filename)
if not os.path.isfile(input_path):
    fail(f"input video not found: {input_path}")

stem, _ = os.path.splitext(video_filename)
output_filename = env("OUTPUT_FILENAME", f"{stem}_redacted.mp4")
output_path = os.path.join(DATA_DIR, output_filename)
if os.path.abspath(output_path) == os.path.abspath(input_path):
    fail("OUTPUT_FILENAME must differ from VIDEO_FILENAME")


def probe_source_video(path):
    """Inspect source video codec and bitrate via ffprobe."""
    info = {
        "codec_name": "h264",
        "bit_rate": 0,
    }
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,bit_rate:format=bit_rate",
            "-of", "json", path
        ], text=True)
        data = json.loads(out)
        streams = data.get("streams", [])
        if streams:
            s = streams[0]
            info["codec_name"] = s.get("codec_name", "h264")
            if s.get("bit_rate"):
                info["bit_rate"] = int(s["bit_rate"])
        if not info["bit_rate"]:
            fmt_br = data.get("format", {}).get("bit_rate")
            if fmt_br:
                info["bit_rate"] = max(500_000, int(fmt_br) - 128_000)
    except Exception:
        pass
    return info


def has_nvenc_support(encoder_name):
    """Test whether the container has NVIDIA NVENC driver capabilities enabled."""
    try:
        res = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-f", "lavfi",
                "-i", "color=c=black:s=256x256:d=0.1:r=10",
                "-c:v", encoder_name, "-f", "null", "-"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return res.returncode == 0
    except Exception:
        return False


def build_encoder_args(src_info, gpu_enabled, crf_val, preset_val):
    """
    Build ffmpeg video encoder arguments that match the source codec and bitrate
    (using NVIDIA NVENC hardware encoding when available) so the output file size
    matches the original video and plays without any splice artifacts.
    """
    codec = src_info["codec_name"]
    src_br = src_info["bit_rate"]

    if codec in ("hevc", "h265"):
        nvenc_enc = "hevc_nvenc"
        cpu_enc = "libx265"
        tag_args = ["-tag:v", "hvc1"]
    else:
        nvenc_enc = "h264_nvenc"
        cpu_enc = "libx264"
        tag_args = []

    use_nvenc = gpu_enabled and has_nvenc_support(nvenc_enc)
    enc = nvenc_enc if use_nvenc else cpu_enc

    args = ["-c:v", enc, "-pix_fmt", "yuv420p"] + tag_args

    # Match source bitrate whenever available (unless user explicitly set a custom CRF > 20)
    if src_br > 0 and (crf_val <= 0 or crf_val == 18):
        target_kbps = max(500, int(src_br / 1000))
        max_kbps = int(target_kbps * 1.25)
        buf_kbps = int(target_kbps * 2.0)
        if use_nvenc:
            args += ["-preset", "p4", "-rc", "vbr", "-b:v", f"{target_kbps}k", "-maxrate", f"{max_kbps}k", "-bufsize", f"{buf_kbps}k"]
        else:
            args += ["-preset", preset_val, "-b:v", f"{target_kbps}k", "-maxrate", f"{max_kbps}k", "-bufsize", f"{buf_kbps}k"]
            if cpu_enc == "libx265":
                args += ["-x265-params", "log-level=error"]
    else:
        effective_crf = crf_val if crf_val > 0 else (26 if cpu_enc == "libx265" else 23)
        if use_nvenc:
            args += ["-preset", "p4", "-cq", str(effective_crf)]
        else:
            args += ["-preset", preset_val, "-crf", str(effective_crf)]
            if cpu_enc == "libx265":
                args += ["-x265-params", "log-level=error"]

    return args, enc


# --- Open video -----------------------------------------------------------
src_info = probe_source_video(input_path)

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    fail(f"could not open video: {input_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps else 0

start_frame = int(START_SECONDS * fps)
end_frame = int(END_SECONDS * fps) if END_SECONDS > 0 else total_frames
if end_frame <= start_frame:
    fail(f"END_SECONDS ({END_SECONDS}) must be after START_SECONDS ({START_SECONDS})")

gpu_active = USE_GPU and torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if gpu_active else "CPU"
enc_args, chosen_encoder = build_encoder_args(src_info, gpu_active, CRF, PRESET)
src_mbps = src_info["bit_rate"] / 1_000_000 if src_info["bit_rate"] else 0

print("=== Video cleanse ===")
print(f"Input      : {input_path}")
print(f"Output     : {output_path}")
print(f"Video      : {width}x{height} @ {fps:.2f}fps, {total_frames} frames ({duration:.1f}s)")
print(f"Source     : codec={src_info['codec_name']}, bitrate={src_mbps:.2f} Mbps")
print(f"Encoder    : {chosen_encoder} (matching source bitrate {src_mbps:.2f} Mbps)")
match_desc = f"matching {MIN_SUBSTRING_LEN}-char substrings: {', '.join(SEARCH_PATTERNS)}" if MIN_SUBSTRING_LEN > 0 else "exact word match"
print(f"Target     : {', '.join(TARGET_WORDS)} ({match_desc})")
print(f"Window     : {start_frame / fps:.1f}s -> {min(end_frame, total_frames) / fps:.1f}s")
print(f"Mode       : {REDACT_MODE}")
print(f"OCR every  : {OCR_INTERVAL} frame(s), GPU={gpu_active} ({gpu_name})")
sys.stdout.flush()

T_START = time.time()


def fmt_dur(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"


def log(msg):
    print(f"[{fmt_dur(time.time() - T_START):>9}] {msg}", flush=True)


window_end = min(end_frame, total_frames)
window_frames = max(0, window_end - start_frame)
total_ocr = (window_frames + OCR_INTERVAL - 1) // OCR_INTERVAL
log("Stage 1/3: loading OCR model")
reader = easyocr.Reader(["en"], gpu=gpu_active, model_storage_directory=MODEL_DIR,
                        download_enabled=True, verbose=False)
log("OCR model loaded")
log(f"Stage 2/3: processing {total_frames} frames; {total_ocr} OCR scans planned "
    f"in the window (frames {start_frame}-{window_end})")

# --- FFmpeg: raw frames in on stdin, original audio copied from source ----
ffmpeg_cmd = [
    "ffmpeg", "-y", "-loglevel", "error",
    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
    "-i", "-",
    "-i", input_path,
    "-map", "0:v:0", "-map", "1:a:0?",
    *enc_args,
    "-c:a", "copy",
    "-movflags", "+faststart",
    "-shortest",
    output_path,
]
ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


def expand_box_to_full_target(bbox, detected_text, matched_pat, img_w, img_h, padding):
    """
    When EasyOCR only detects a partial substring of a target word (e.g. 'nifo' or
    'aedenstani' inside 'braedenstanifo'), extrapolate the bounding box left and right
    based on the estimated character width so the unredacted start/end of the word is
    fully covered.
    """
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1, x2 = float(min(xs)), float(max(xs))
    y1, y2 = float(min(ys)), float(max(ys))
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    text_lower = detected_text.lower()
    pat_idx_in_det = text_lower.find(matched_pat)

    # Find which target word this pattern belongs to
    parent_word = next((w for w in TARGET_WORDS if matched_pat in w), matched_pat)
    pat_idx_in_target = parent_word.find(matched_pat)

    if pat_idx_in_det >= 0 and pat_idx_in_target >= 0:
        # Estimate average single-character width from bounding box and height
        det_len = max(1, len(detected_text.strip()))
        char_w = max(box_w / det_len, box_h * 0.55)

        # How many characters of parent_word extend to the left/right of detected_text
        missing_left_chars = max(0, pat_idx_in_target - pat_idx_in_det)
        chars_after_pat_in_target = len(parent_word) - (pat_idx_in_target + len(matched_pat))
        chars_after_pat_in_det = len(text_lower) - (pat_idx_in_det + len(matched_pat))
        missing_right_chars = max(0, chars_after_pat_in_target - chars_after_pat_in_det)

        # Add generous safety margin (+1.5 chars on each missing side) so partial reads are covered
        if missing_left_chars > 0:
            x1 -= (missing_left_chars + 1.5) * char_w
        if missing_right_chars > 0:
            x2 += (missing_right_chars + 1.5) * char_w

    return (
        max(0, int(x1) - padding),
        max(0, int(y1) - padding),
        min(img_w, int(x2) + padding),
        min(img_h, int(y2) + padding),
    )


def redact(frame, box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    if REDACT_MODE == "block":
        cv2.rectangle(frame, (x1, y1), (x2, y2), BLOCK_COLOUR, thickness=-1)
    else:
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (BLUR_STRENGTH, BLUR_STRENGTH), 0)


frame_idx = 0
active_boxes = []
frames_since_detection = 0
detections = 0
redacted_frames = 0
last_report = time.time()
ocr_done = 0
ocr_time = 0.0
other_time = 0.0
other_frames = 0
stage = None


def report():
    pct = 100 * frame_idx / total_frames if total_frames else 0
    avg_ocr = ocr_time / ocr_done if ocr_done else None
    avg_other = other_time / other_frames if other_frames else 0
    remaining_ocr = max(0, total_ocr - ocr_done)
    remaining_frames = max(0, total_frames - frame_idx)
    if remaining_ocr and avg_ocr is None:
        eta = "calculating"
    else:
        eta = fmt_dur(remaining_ocr * (avg_ocr or 0) + remaining_frames * avg_other)
    parts = [
        f"{pct:5.1f}% | frame {frame_idx}/{total_frames}",
        f"video {frame_idx / fps:.1f}s/{duration:.1f}s",
    ]
    if total_ocr:
        wpct = 100 * ocr_done / total_ocr
        parts.append(f"window {wpct:.1f}% ({ocr_done}/{total_ocr} OCR scans)")
    if avg_ocr:
        parts.append(f"{avg_ocr:.2f}s/scan")
    parts.append(f"hits {detections}")
    parts.append(f"ETA {eta}")
    log(" | ".join(parts))


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_t = time.time()

    in_window = start_frame <= frame_idx < end_frame

    new_stage = ("before window" if frame_idx < start_frame
                 else "scanning window" if in_window else "after window")
    if new_stage != stage:
        stage = new_stage
        log(f"-> {stage} (video {frame_idx / fps:.1f}s, frame {frame_idx})")

    if in_window:
        if (frame_idx - start_frame) % OCR_INTERVAL == 0:
            detected_now = []
            matched_samples = []
            ocr_t = time.time()
            results = reader.readtext(frame)
            ocr_time += time.time() - ocr_t
            ocr_done += 1
            for bbox, text, prob in results:
                if prob < MIN_CONFIDENCE:
                    continue
                text_lower = text.lower()
                matched_pat = next((pat for pat in SEARCH_PATTERNS if pat in text_lower), None)
                if matched_pat:
                    expanded_box = expand_box_to_full_target(
                        bbox, text, matched_pat, width, height, PADDING
                    )
                    detected_now.append(expanded_box)
                    matched_samples.append(f"'{text}' (matched '{matched_pat}')")
            if detected_now:
                if not active_boxes:
                    log(f"   found {', '.join(matched_samples)} at video {frame_idx / fps:.2f}s "
                        f"({len(detected_now)} box(es))")
                active_boxes = detected_now
                frames_since_detection = 0
                detections += 1
            else:
                frames_since_detection += OCR_INTERVAL
                if frames_since_detection > HOLD_FRAMES:
                    active_boxes = []

        for box in active_boxes:
            redact(frame, box)
        if active_boxes:
            redacted_frames += 1
    elif active_boxes:
        active_boxes = []

    try:
        ffmpeg.stdin.write(frame.tobytes())
    except BrokenPipeError:
        fail("ffmpeg exited unexpectedly")

    frame_idx += 1

    now = time.time()
    if not (in_window and (frame_idx - 1 - start_frame) % OCR_INTERVAL == 0):
        other_time += now - frame_t
        other_frames += 1
    if now - last_report >= PROGRESS_SECONDS:
        report()
        last_report = now

report()
cap.release()
log("Stage 3/3: finalising encode")
ffmpeg.stdin.close()
if ffmpeg.wait() != 0:
    fail("ffmpeg failed to encode the output")

out_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
in_size_mb = os.path.getsize(input_path) / (1024 * 1024)
log(f"Done in {fmt_dur(time.time() - T_START)}. OCR scans: {ocr_done}, "
    f"hits: {detections}, frames redacted: {redacted_frames}")
print(f"Saved: {output_path} ({out_size_mb:.1f} MB vs source {in_size_mb:.1f} MB)")
