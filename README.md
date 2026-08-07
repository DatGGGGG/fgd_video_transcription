# FGD Video Transcription Pipeline

This project is a practical workflow for transcribing a focus group discussion (FGD) video where pure audio diarization is not reliable enough.

The core idea is:

1. Use audio-only transcription to get the best Vietnamese text and timestamped speech lines.
2. Convert those lines into segments with begin and end timestamps.
3. Sample video frames around each segment.
4. Use a vision-capable LLM to answer only: who appears to be talking in these frames?
5. Fuse the audio transcript and visual speaker evidence.
6. Flag ambiguous, overlapping, or low-confidence sections for manual review.

This is designed for a 8-10 person room setup where speaker labels from audio alone can drift across chunks.

## Why This Approach

Audio-only transcription is better at words than the video model in this setup. Video is still valuable, but mainly for visual attribution: mouth movement, gaze, posture, and the numbered seating layout.

The video has useful information that audio diarization does not have:

- which participant is moving their mouth,
- who is gesturing while speaking,
- who is being addressed by the moderator,
- whether a segment is likely cross-talk.

The recommended output should not pretend to be perfect. It should produce a best speaker guess plus review flags.

## First Step: Model Bake-off

Do not lock the full pipeline to one STT/diarization model until a real FGD sample has been tested.

Run the same 5-10 minute clip through:

- OpenAI `gpt-4o-transcribe-diarize` via `/audio/internal/transcriptions`
- Gemini audio understanding via `/models/{model}:generateContent`
- optionally Gemini video understanding on the same clip

```powershell
python scripts/fgd_pipeline.py bakeoff --video input.mp4 --out-dir outputs/bakeoff --start 0 --duration 600
```

To include Gemini video:

```powershell
python scripts/fgd_pipeline.py bakeoff --video input.mp4 --out-dir outputs/bakeoff --start 0 --duration 600 --include-gemini-video
```

The bake-off writes:

- `outputs/bakeoff/openai_diarize.json`
- `outputs/bakeoff/gemini_audio.json`
- `outputs/bakeoff/gemini_video.json` when enabled
- `outputs/bakeoff/bakeoff_comparison.md`

Score the outputs manually for speaker count, label consistency, timestamp usefulness, overlap handling, and transcript readability.

## Recommended Workflow After Bake-off

### 1. Prepare the video

Install `ffmpeg` and make sure it is available on your `PATH`.

Recommended audio extraction:

```powershell
ffmpeg -y -i input.mp4 -vn -ac 1 -ar 16000 outputs/audio.wav
```

Mono 16 kHz is usually enough for speech transcription.

### 2. Create a participant map

Create `participants.json` from the numbered seating layout visible in the room.

```json
[
  {
    "id": "P1",
    "seat_label": "1",
    "display_name": "Participant 1",
    "visual_description": "left side near bottom, red sweater",
    "role": "participant"
  },
  {
    "id": "MOD",
    "seat_label": "moderator",
    "display_name": "Moderator",
    "visual_description": "bottom center, back to camera, laptop",
    "role": "moderator"
  }
]
```

The more concrete the visual descriptions are, the better the vision pass can map a visible speaker to a stable ID.

### 3. Configure Compass Gateway

This project calls ChatGPT models through Compass Gateway using the same environment pattern as `vn_competitor_event_data_system`.

Create `.env` from `.env.example` and fill in your Compass key:

```bash
cp .env.example .env
nano .env
```

The CLI loads `.env` automatically. You can also export variables manually:

```powershell
$env:OPENAI_API_KEY="replace_me"
$env:OPENAI_BASE_URL="https://compass.llm.shopee.io/compass-api/v1"
$env:OPENAI_PROVIDER="OpenAI"
$env:OPENAI_TRANSCRIPTION_MODEL="gpt-4o-transcribe-diarize"
$env:OPENAI_TRANSCRIPTION_ENDPOINT="/audio/internal/transcriptions"
$env:GEMINI_MODEL="gemini-2.5-flash"
$env:OPENAI_VISION_MODEL="gpt-5.4-mini"
$env:OPENAI_TIMEOUT_SECONDS="300"
$env:OPENAI_MAX_RETRIES="3"
```

`OPENAI_MODEL` is also supported as a general fallback model.

### 4. Transcribe audio

Use the bake-off audio path that performed best. The current recommended route is Gemini audio via Compass. The prompt asks for one continuous speaking turn per line:

```text
[HH:MM:SS.mmm - HH:MM:SS.mmm] Speaker A: text
```

If the model only returns one timestamp per line, the helper can still infer each line's end from the next later timestamp.

### 5. Convert Gemini audio output into segments

```bash
python3 scripts/fgd_pipeline.py segmentize-gemini \
  --gemini-json outputs/bakeoff/gemini_audio.json \
  --out outputs/bakeoff/audio_segments.json \
  --clip-duration 285.37
```

### 6. Sample frames around each speech segment

For each speech segment, sample up to 3 frames by default. The sampler extracts audio inside that segment, scores short windows by loudness, and picks the strongest speech moments with spacing between them. This avoids grabbing silent pauses inside a broad transcript segment.

Very short segments may only need one middle frame. Use the sample clip when the transcript timestamps are relative to the sample clip.

```bash
python3 scripts/fgd_pipeline.py sample-frames \
  --video outputs/bakeoff/sample_clip.mp4 \
  --diarized outputs/bakeoff/audio_segments.json \
  --frames-dir outputs/bakeoff/frames_3 \
  --manifest outputs/bakeoff/frame_manifest_3.json \
  --max-frames 3
```

### 7. Assign visible speakers

For each segment, send the sampled frames, participant map, and segment text to a vision-capable model. Ask for strict JSON:

```json
{
  "segment_id": "seg_001",
  "visible_speaker_id": "P4",
  "confidence": 0.74,
  "evidence": "P4 appears to be facing the table and mouth is open in the sampled speech frames.",
  "flags": ["low_visual_certainty"]
}
```

Before calling the model, run an estimate:

```bash
python3 scripts/fgd_pipeline.py assign-speakers \
  --manifest outputs/bakeoff/frame_manifest_3.json \
  --participants examples/participants.example.json \
  --out outputs/bakeoff/vision_segments.json \
  --estimate-only
```

The real run prints the same estimate and asks you to type `YES` before any AI calls. It also logs each segment and saves progress after every segment, so you can stop and resume.

```bash
python3 scripts/fgd_pipeline.py assign-speakers \
  --manifest outputs/bakeoff/frame_manifest_3.json \
  --participants examples/participants.example.json \
  --out outputs/bakeoff/vision_segments.json
```

### 8. Fuse and flag

Use the audio speaker label as a temporary turn grouping signal, not as the final participant identity.

Flag for manual review when:

- audio speaker changes but visual speaker does not,
- visual confidence is low,
- multiple people appear to speak,
- segment is shorter than 1 second,
- segment overlaps with another segment,
- transcript text contains interruption markers,
- video frame does not show a clear face/mouth.

## Files

- `scripts/fgd_pipeline.py` - CLI helper for extraction, frame sampling, fusion, and review output.
- `examples/participants.example.json` - starter participant map.
- `docs/compass_arsenal.md` - notes from the Compass Gateway docs about usable model routes.

## Minimal Command Flow

```powershell
python scripts/fgd_pipeline.py extract-audio --video input.mp4 --audio outputs/audio.wav
python scripts/fgd_pipeline.py bakeoff --video input.mp4 --out-dir outputs/bakeoff --skip-openai
python scripts/fgd_pipeline.py segmentize-gemini --gemini-json outputs/bakeoff/gemini_audio.json --out outputs/bakeoff/audio_segments.json
python scripts/fgd_pipeline.py sample-frames --video outputs/bakeoff/sample_clip.mp4 --diarized outputs/bakeoff/audio_segments.json --frames-dir outputs/frames_3 --manifest outputs/frame_manifest_3.json --max-frames 3
python scripts/fgd_pipeline.py assign-speakers --manifest outputs/frame_manifest_3.json --participants participants.json --out outputs/vision_segments.json --estimate-only
python scripts/fgd_pipeline.py assign-speakers --manifest outputs/frame_manifest_3.json --participants participants.json --out outputs/vision_segments.json
python scripts/fgd_pipeline.py fuse --diarized outputs/bakeoff/audio_segments.json --vision outputs/vision_segments.json --participants participants.json --out outputs/final_transcript.md
```

The `transcribe` and `assign-speakers` steps are intentionally model/API-configurable because access and preferred model names may differ by account.

## Setup Notes

Python dependencies:

```powershell
pip install -r requirements.txt
```

The current script uses only Python standard library modules, so `requirements.txt` is intentionally empty except for comments.

This repository does not bundle `ffmpeg`. Install it separately and confirm `ffmpeg -version` works before running video/audio commands.

## WSL

For Ubuntu/WSL usage, see `docs/wsl.md`.

Quick command once `Ubuntu-D` exists:

```powershell
.\scripts\run_bakeoff_wsl.ps1 -Distro Ubuntu-D -Video inputs\fgd_sample_hbs_5_min_clean.mp4 -OutDir outputs\bakeoff -Duration 600
```
