# FGD Pipeline Handoff

Use this when continuing the project from another machine or another Codex session.

## Goal

Build a practical FGD video transcription workflow for a multi-person room recording where audio diarization alone is unreliable.

The chosen architecture is:

1. Use the audio-only Gemini path through Compass for Vietnamese transcript quality.
2. Convert transcript lines into timestamped segments with begin/end times.
3. Extract video frames per segment at moments where the segment audio is loudest, not arbitrary early/mid/late positions.
4. Use a vision-capable model only for the narrow task: identify who appears to be talking in the sampled frames.
5. Fuse the audio transcript text with visual speaker assignment and flag uncertain segments for manual review.

## Current Repo State

The project is pushed to:

```text
https://github.com/DatGGGGG/fgd_video_transcription.git
```

Important local-only files are intentionally ignored:

- `.env`
- `.venv/`
- `outputs/`
- `inputs/*.mp4`
- extracted frames/audio/video artifacts

On a new machine, create your own `.env` from `.env.example`.

## Key Commands

From WSL:

```bash
cd /mnt/d/Coding/fgd_video_transcription
source .venv/bin/activate
```

Run audio-only bakeoff/transcription:

```bash
python3 scripts/fgd_pipeline.py bakeoff \
  --video inputs/fgd_sample_hbs_5_min_clean.mp4 \
  --out-dir outputs/bakeoff \
  --duration 300 \
  --skip-openai
```

Convert Gemini output to segment JSON:

```bash
python3 scripts/fgd_pipeline.py segmentize-gemini \
  --gemini-json outputs/bakeoff/gemini_audio.json \
  --out outputs/bakeoff/audio_segments.json \
  --clip-duration 285.37
```

Extract frames from the loudest speech moments inside each segment:

```bash
python3 scripts/fgd_pipeline.py sample-frames \
  --video outputs/bakeoff/sample_clip.mp4 \
  --diarized outputs/bakeoff/audio_segments.json \
  --frames-dir outputs/bakeoff/frames_audiopeak_3 \
  --manifest outputs/bakeoff/frame_manifest_audiopeak_3.json \
  --max-frames 3
```

Estimate vision speaker-assignment cost before model calls:

```bash
python3 scripts/fgd_pipeline.py assign-speakers \
  --manifest outputs/bakeoff/frame_manifest_audiopeak_3.json \
  --participants examples/participants.example.json \
  --out outputs/bakeoff/vision_segments.json \
  --estimate-only
```

Run vision speaker assignment after reviewing the estimate:

```bash
python3 scripts/fgd_pipeline.py assign-speakers \
  --manifest outputs/bakeoff/frame_manifest_audiopeak_3.json \
  --participants examples/participants.example.json \
  --out outputs/bakeoff/vision_segments.json
```

The real run prints the estimate, asks for `YES`, logs progress per segment, and saves progress after every segment. Re-running resumes by skipping already completed segment IDs.

Fuse final transcript:

```bash
python3 scripts/fgd_pipeline.py fuse \
  --diarized outputs/bakeoff/audio_segments.json \
  --vision outputs/bakeoff/vision_segments.json \
  --participants examples/participants.example.json \
  --out outputs/bakeoff/final_transcript.md
```

## Known Findings

- OpenAI `gpt-4o-transcribe-diarize` through Compass failed for this setup with gateway/ASR errors. Do not block on it.
- Gemini audio gave the best transcript quality in the initial bakeoff.
- Gemini video transcription degraded word quality, so video should not be used for full transcription.
- The vision step should only answer who is visibly talking for a timestamped segment.
- Windows PowerShell may display Vietnamese UTF-8 as mojibake. WSL `cat`/`less`, VS Code, or Python UTF-8 reads are fine.

## Operational Guardrails

- Do not commit `.env`, clips, frames, outputs, or API results with sensitive data.
- Before AI-calling commands, run `--estimate-only`.
- For long `assign-speakers` jobs, output is checkpointed after each segment.
- Use WSL for ffmpeg work.

