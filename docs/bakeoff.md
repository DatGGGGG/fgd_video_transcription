# Bake-off Harness

The first engineering milestone is a model bake-off on a real 5-10 minute FGD clip.

## Why

The project has two viable first-pass transcription options:

- OpenAI `gpt-4o-transcribe-diarize`, purpose-built for diarized audio transcription.
- Gemini 2.5/3 Flash or Pro, prompt-driven multimodal transcription with audio or video input.

Neither should be assumed best for 10-person FGD audio. The bake-off creates side-by-side artifacts for human comparison before the rest of the pipeline is optimized around one route.

## Command

```powershell
python scripts/fgd_pipeline.py bakeoff --video input.mp4 --out-dir outputs/bakeoff --start 0 --duration 600
```

Optional video route:

```powershell
python scripts/fgd_pipeline.py bakeoff --video input.mp4 --out-dir outputs/bakeoff --start 0 --duration 600 --include-gemini-video
```

## Environment

```powershell
$env:OPENAI_API_KEY="replace_me"
$env:OPENAI_BASE_URL="https://compass.llm.shopee.io/compass-api/v1"
$env:OPENAI_PROVIDER="OpenAI"
$env:OPENAI_TRANSCRIPTION_MODEL="gpt-4o-transcribe-diarize"
$env:OPENAI_TRANSCRIPTION_ENDPOINT="/audio/internal/transcriptions"
$env:GEMINI_MODEL="gemini-2.5-flash"
```

## Outputs

- `sample_clip.mp4` - clipped source media.
- `sample_audio.mp3` - mono 16 kHz audio extracted from the sample.
- `openai_diarize.json` - raw Compass/OpenAI diarized transcript.
- `gemini_audio.json` - raw Compass/Gemini audio transcript.
- `gemini_video.json` - raw Compass/Gemini video transcript when `--include-gemini-video` is set.
- `bakeoff_comparison.md` - human-readable comparison file.
- `bakeoff_summary.json` - metadata and parsed text from all routes.

## Scoring Rubric

Score each output by:

- Speaker count accuracy.
- Speaker label consistency.
- Timestamp usefulness.
- Overlap/crosstalk handling.
- Transcript readability.

Use the winner to decide which STT/diarization function should become the primary pipeline path.

