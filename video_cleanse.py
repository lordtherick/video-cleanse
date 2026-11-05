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
  CRF              x264 quality, lower = better/bigger              (default 18)
  PRESET           x264 preset                                      (default fast)
  USE_GPU          true to use CUDA if available                    (default false)

Frames outside START_SECONDS..END_SECONDS are passed through untouched, so the
output is always the full-length video with the original audio.
"""
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

try:
    START_SECONDS = float(env("START_SECONDS", 0))
    END_SECONDS = float(env("END_SECONDS", 0))
    OCR_INTERVAL = max(1, int(env("OCR_INTERVAL", 5)))
    HOLD_FRAMES = max(0, int(env("HOLD_FRAMES", 10)))
    PADDING = max(0, int(env("PADDING", 10)))
    MIN_CONFIDENCE = float(env("MIN_CONFIDENCE", 0.3))
    BLUR_STRENGTH = int(env("BLUR_STRENGTH", 51))
    CRF = int(env("CRF", 18))
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

# --- Open video -----------------------------------------------------------
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

print("=== Video cleanse ===")
print(f"Input      : {input_path}")
print(f"Output     : {output_path}")
print(f"Video      : {width}x{height} @ {fps:.2f}fps, {total_frames} frames ({duration:.1f}s)")
print(f"Target     : {', '.join(TARGET_WORDS)}")
print(f"Window     : {start_frame / fps:.1f}s -> {min(end_frame, total_frames) / fps:.1f}s")
print(f"Mode       : {REDACT_MODE}")
print(f"OCR every  : {OCR_INTERVAL} frame(s), GPU={USE_GPU}")
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
log(f"Stage 1/3: loading OCR model")
reader = easyocr.Reader(["en"], gpu=USE_GPU, model_storage_directory=MODEL_DIR,
                        download_enabled=True, verbose=False)
log(f"OCR model loaded")
log(f"Stage 2/3: processing {total_frames} frames; {total_ocr} OCR scans planned "
    f"in the window (frames {start_frame}-{window_end})")

# --- FFmpeg: raw frames in on stdin, original audio copied from source ----
ffmpeg_cmd = [
    "ffmpeg", "-y", "-loglevel", "error",
    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
    "-i", "-",
    "-i", input_path,
    "-map", "0:v:0", "-map", "1:a:0?",
    "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET, "-pix_fmt", "yuv420p",
    "-c:a", "copy",
    "-movflags", "+faststart",
    "-shortest",
    output_path,
]
ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


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
t0 = time.time()
last_report = t0
ocr_done = 0
ocr_time = 0.0       # seconds spent inside OCR
other_time = 0.0     # seconds spent decoding/redacting/encoding
other_frames = 0
stage = None


def report(now):
    pct = 100 * frame_idx / total_frames if total_frames else 0
    avg_ocr = ocr_time / ocr_done if ocr_done else None
    avg_other = other_time / other_frames if other_frames else 0
    remaining_ocr = total_ocr - ocr_done
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
            ocr_t = time.time()
            results = reader.readtext(frame)
            ocr_time += time.time() - ocr_t
            ocr_done += 1
            for bbox, text, prob in results:
                if prob < MIN_CONFIDENCE:
                    continue
                if any(w in text.lower() for w in TARGET_WORDS):
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    detected_now.append((
                        max(0, int(min(xs)) - PADDING),
                        max(0, int(min(ys)) - PADDING),
                        min(width, int(max(xs)) + PADDING),
                        min(height, int(max(ys)) + PADDING),
                    ))
            if detected_now:
                if not active_boxes:
                    log(f"   found '{target_raw}' at video {frame_idx / fps:.2f}s "
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
        report(now)
        last_report = now

report(time.time())
cap.release()
log("Stage 3/3: finalising encode")
ffmpeg.stdin.close()
if ffmpeg.wait() != 0:
    fail("ffmpeg failed to encode the output")

log(f"Done in {fmt_dur(time.time() - T_START)}. OCR scans: {ocr_done}, "
    f"hits: {detections}, frames redacted: {redacted_frames}")
print(f"Saved: {output_path}")
