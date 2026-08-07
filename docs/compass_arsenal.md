# Compass Arsenal Notes

These notes are based on the local Compass documentation snapshots in `C:\Users\VEE0634\Downloads`.

## OpenAI API Proxy

Base URL:

```text
https://compass.llm.shopee.io/compass-api/v1
```

Common headers:

```text
Authorization: Bearer <Your Compass API-KEY>
Content-Type: application/json
Provider: OpenAI
```

Relevant endpoints for this FGD project:

- `/responses` - structured reasoning and vision-supported workflows when using OpenAI-compatible models.
- `/chat/completions` - fallback route for chat-style models.
- `/audio/transcriptions` - standard transcription.
- `/audio/internal/transcriptions` - diarizing transcription path shown in the Compass/OpenAI docs and pricing screenshot.
- `/audio/translations` - speech translation.
- `/files` and `/batches` - possible later optimization for batch processing.

For FGD transcription, default to `/audio/internal/transcriptions` with `gpt-4o-transcribe-diarize`, because the diarize model is listed under the internal transcription pricing/endpoint.

## Gemini API Proxy

Gemini is available through Compass with:

- direct Gemini-style endpoints such as `/models/{MODEL_ID}:generateContent`,
- OpenAI-compatible `/chat/completions`,
- context caching via `/cachedContents`.

Useful models mentioned in the docs:

- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

Capabilities called out in the docs include text/code generation, image understanding, video understanding, audio understanding, document understanding, function calling, structured output, batch prediction, and thinking.

FGD use:

- Strong candidate for the vision speaker-mapping pass.
- Good fallback if an OpenAI vision model struggles with the room camera angle.
- Context caching could help if we repeatedly send many frames plus the same participant/seat map.

## Claude API Proxy

Claude is available through Compass using Anthropic-style `/messages`.

FGD use:

- Good candidate for transcript cleanup, review-flag reasoning, and final narrative cleanup.
- Less central for diarization because the pipeline needs timestamped audio segmentation first.

## Open-Source Model Support

Compass in-house/open-source models include:

- Compass Max
- Compass V2
- Compass Code
- Compass LLVM / Compass-VL

The docs describe Compass LLVM as vision-enhanced. This may be worth testing on speaker mapping if corporate policy prefers in-house models or if cost is important.

## Google Text To Speech Snapshot

The file named `Perplexity.html` appears to contain the Google Text to Speech page, not Perplexity. It lists Gemini TTS and Chirp voices. This is not needed for FGD transcription unless we later want transcript readout or synthetic QA audio.

## Practical Model Routing For FGD

Recommended first pass:

1. Bake-off: compare OpenAI `gpt-4o-transcribe-diarize` against Gemini audio, and optionally Gemini video.
2. Audio segmentation/transcription: choose the bake-off winner. Do not assume OpenAI diarization wins by default.
3. Visual speaker mapping: Gemini `gemini-2.5-pro` or `gemini-2.5-flash`, or OpenAI-compatible vision via `/responses`.
4. Repair and review flags: cheaper strong text model such as `gpt-5.4-mini`, or Claude if text reasoning quality is better in practice.

The pipeline should keep model names and endpoints configurable because Compass supports several provider routes with slightly different endpoint shapes.
