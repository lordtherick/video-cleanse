# Video Cleanse (Docker)

Finds a word on screen with OCR (EasyOCR) and blurs or blocks it out.
The output is the full-length video with the original audio, saved next to the input.

## Files

| File | Purpose |
|---|---|
| `video_cleanse.py` | Container script (parameterised version of `video-cleanse.py`) |
| `Dockerfile` | Python 3.14 + ffmpeg + CPU PyTorch + EasyOCR, with OCR models baked in |
| `docker-compose.yml` | Wrapper that mounts the video folder and reads `.env` |
| `.env.example` | All settings - copy to `.env` before running |
| `.dockerignore` | Keeps large videos out of the build context |

## Quick start

```bash
# 1. cp .env.example .env, then edit .env (video name, word, time window, blur/block)
# 2. Build once (takes a few minutes - downloads PyTorch and the OCR models)
docker compose build
# 3. Run
docker compose run --rm video-cleanse
```

The output is written to `VIDEO_DIR` as `<name>_redacted.mp4` unless `OUTPUT_FILENAME` is set.

## Settings (.env)

| Variable | Default | Meaning |
|---|---|---|
| `VIDEO_FILENAME` | *(required)* | Video to review, inside `VIDEO_DIR` |
| `TARGET_WORD` | *(required)* | Word to redact, case-insensitive. Comma-separate several: `brae,braes` |
| `START_SECONDS` | `0` | Start of the section to review |
| `END_SECONDS` | end | End of the section to review (blank or `0` = to the end) |
| `REDACT_MODE` | `blur` | `blur` or `block` |
| `VIDEO_DIR` | `.` | Folder on the host holding the video |
| `OUTPUT_FILENAME` | `<name>_redacted.mp4` | Output file name |
| `OCR_INTERVAL` | `5` | OCR every N frames. Lower = catches brief text, slower |
| `HOLD_FRAMES` | `10` | Keep the box N frames after last seen (stops flicker) |
| `PADDING` | `10` | Pixels added around each detected word |
| `MIN_CONFIDENCE` | `0.3` | OCR confidence threshold |
| `BLUR_STRENGTH` | `51` | Blur kernel (odd); bigger = heavier |
| `BLOCK_COLOUR` | `0,0,0` | R,G,B fill for block mode |
| `CRF` / `PRESET` | `18` / `fast` | x264 quality / speed |
| `PROGRESS_SECONDS` | `10` | Seconds between progress log lines |
| `USE_GPU` | `false` | CUDA (needs a GPU build, see below) |

Any setting can be overridden for a single run without editing `.env`:

```bash
docker compose run --rm -e REDACT_MODE=block -e START_SECONDS=30 -e END_SECONDS=95 video-cleanse
```

## Notes

- Only frames between `START_SECONDS` and `END_SECONDS` are scanned. Frames outside that window are copied through unchanged, but the whole video is still re-encoded once.
- OCR on CPU is the slow part. Narrowing the window, or raising `OCR_INTERVAL`, speeds things up a lot.
- Matching is a substring match on each OCR text line, so `brae` also catches "Brae's". When the word is part of a longer line, the whole line is redacted.
- **NVIDIA GPU host:** `docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121`, set `USE_GPU=true`, and add a GPU device reservation to the compose service.
- Always check the output by eye; OCR can miss stylised, tiny or fast-moving text.
