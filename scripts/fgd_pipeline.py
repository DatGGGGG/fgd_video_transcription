#!/usr/bin/env python
"""Utilities for an audio + vision FGD transcription workflow."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


REVIEW_FLAGS = {
    "low_visual_confidence",
    "audio_visual_mismatch",
    "short_segment",
    "possible_overlap",
    "no_visual_assignment",
}


@dataclass(frozen=True)
class GatewayConfig:
    api_key: str
    base_url: str
    provider: str
    model: str
    transcription_model: str
    transcription_endpoint: str
    gemini_model: str
    vision_model: str
    timeout_seconds: int
    max_retries: int


@dataclass
class Segment:
    id: str
    start: float
    end: float
    text: str
    audio_speaker: str | None


TRANSCRIPT_LINE_RE = re.compile(
    r"^\[(?P<start>\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)"
    r"(?:\s*[-–]\s*(?P<end>\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?))?\]\s*"
    r"(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)


def normalize_gateway_base_url(value: str) -> str:
    text = value.strip().rstrip("/")
    for suffix in ("/responses", "/audio/transcriptions"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or name in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value


def load_gateway_config() -> GatewayConfig:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY before calling Compass Gateway.")

    model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini").strip() or "gpt-5.4-mini"
    return GatewayConfig(
        api_key=api_key,
        base_url=normalize_gateway_base_url(
            os.environ.get("OPENAI_BASE_URL", "https://compass.llm.shopee.io/compass-api/v1")
        ),
        provider=os.environ.get("OPENAI_PROVIDER", "OpenAI").strip() or "OpenAI",
        model=model,
        transcription_model=os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-transcribe-diarize").strip()
        or "gpt-4o-transcribe-diarize",
        transcription_endpoint=os.environ.get("OPENAI_TRANSCRIPTION_ENDPOINT", "/audio/internal/transcriptions").strip()
        or "/audio/internal/transcriptions",
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash",
        vision_model=os.environ.get("OPENAI_VISION_MODEL", model).strip() or model,
        timeout_seconds=max(30, int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "300").strip() or "300")),
        max_retries=max(1, int(os.environ.get("OPENAI_MAX_RETRIES", "3").strip() or "3")),
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def gateway_headers(config: GatewayConfig, content_type: str, *, include_provider: bool = True) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": content_type,
    }
    if include_provider:
        headers["Provider"] = config.provider
    return headers


def gateway_request_json(
    config: GatewayConfig,
    path: str,
    payload: dict[str, Any],
    *,
    include_provider: bool = True,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{config.base_url}{path}",
        data=body,
        headers=gateway_headers(config, "application/json", include_provider=include_provider),
        method="POST",
    )
    return gateway_request(config, req)


def gateway_request(config: GatewayConfig, req: request.Request) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            with request.urlopen(req, timeout=config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError(f"Unexpected gateway response type: {type(payload)}")
            return payload
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(f"Gateway request failed: status={exc.code}, body={body[:1000]}") from exc
        except TimeoutError as exc:
            last_error = exc
        except error.URLError as exc:
            last_error = exc

        if attempt < config.max_retries:
            time.sleep(min(2 ** (attempt - 1), 8))

    assert last_error is not None
    raise RuntimeError(
        "Gateway request failed after retries: "
        f"timeout={config.timeout_seconds}s retries={config.max_retries} error={last_error}"
    ) from last_error


def is_chunking_strategy_error(exc: Exception) -> bool:
    text = str(exc)
    return "chunking_strategy" in text and "diarization" in text


def build_multipart_form(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    boundary = f"----fgdBoundary{int(time.time() * 1000)}"
    chunks: list[bytes] = []

    def add(value: str) -> None:
        chunks.append(value.encode("utf-8"))

    for name, value in fields.items():
        add(f"--{boundary}\r\n")
        add(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        add(f"{value}\r\n")

    for name, path in files.items():
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        add(f"--{boundary}\r\n")
        add(f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n')
        add(f"Content-Type: {mime}\r\n\r\n")
        chunks.append(path.read_bytes())
        add("\r\n")

    add(f"--{boundary}--\r\n")
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def run_ffmpeg(args: list[str]) -> None:
    try:
        subprocess.run(["ffmpeg", "-y", *args], check=True)
    except FileNotFoundError:
        raise SystemExit("ffmpeg is not installed or is not on PATH.")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg failed with exit code {exc.returncode}.")


def read_segment_pcm(video: Path, segment: Segment, *, sample_rate: int) -> list[int]:
    duration = max(0.05, segment.end - segment.start)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{segment.start:.3f}",
        "-i",
        str(video),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise SystemExit("ffmpeg is not installed or is not on PATH.")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg audio extraction failed for {segment.id}: {stderr[:500]}") from exc
    pcm = completed.stdout
    if len(pcm) < 2:
        return []
    sample_count = len(pcm) // 2
    return list(struct.unpack(f"<{sample_count}h", pcm[: sample_count * 2]))


def trim_video(video: Path, out: Path, *, start: float, duration: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(out),
    ])


def extract_transcription_audio(video: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        "64k",
        str(out),
    ])


def parse_segments(diarized: dict[str, Any]) -> list[Segment]:
    raw_segments = diarized.get("segments", [])
    segments: list[Segment] = []
    for i, item in enumerate(raw_segments):
        start = float(item.get("start", 0))
        end = float(item.get("end", start))
        segments.append(
            Segment(
                id=str(item.get("id") or f"seg_{i + 1:04d}"),
                start=start,
                end=end,
                text=str(item.get("text", "")).strip(),
                audio_speaker=item.get("speaker"),
            )
        )
    return segments


def parse_timecode(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timecode: {value}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_transcript_lines(text: str, *, default_tail_seconds: float, clip_duration: float | None) -> list[Segment]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = TRANSCRIPT_LINE_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "start": parse_timecode(match.group("start")),
                "end": parse_timecode(match.group("end")) if match.group("end") else None,
                "speaker": match.group("speaker").strip(),
                "text": match.group("text").strip(),
            }
        )

    segments: list[Segment] = []
    for i, row in enumerate(rows):
        start = float(row["start"])
        end = row["end"]
        if end is None:
            next_starts = [float(next_row["start"]) for next_row in rows[i + 1 :] if float(next_row["start"]) > start]
            if next_starts:
                end = next_starts[0]
            elif clip_duration is not None:
                end = clip_duration
            else:
                end = start + default_tail_seconds
        end = max(float(end), start + 0.25)
        segments.append(
            Segment(
                id=f"seg_{i + 1:04d}",
                start=start,
                end=end,
                text=str(row["text"]),
                audio_speaker=str(row["speaker"]),
            )
        )
    return segments


def segments_to_payload(segments: list[Segment], *, source: str, note: str) -> dict[str, Any]:
    return {
        "source": source,
        "note": note,
        "segments": [
            {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "speaker": segment.audio_speaker,
                "text": segment.text,
            }
            for segment in segments
        ],
    }


def extract_audio(args: argparse.Namespace) -> None:
    extract_transcription_audio(args.video, args.audio)


def segmentize_gemini(args: argparse.Namespace) -> None:
    payload = load_json(args.gemini_json)
    text = extract_gemini_text(payload)
    segments = parse_transcript_lines(
        text,
        default_tail_seconds=args.default_tail_seconds,
        clip_duration=args.clip_duration,
    )
    if not segments:
        raise SystemExit(f"No timestamped transcript lines found in {args.gemini_json}.")
    note = (
        "End timestamps were parsed when present. Lines with only a begin timestamp use the next later "
        "line start as their end; the last line uses --clip-duration or --default-tail-seconds."
    )
    save_json(args.out, segments_to_payload(segments, source=str(args.gemini_json), note=note))


def fallback_sample_times(segment: Segment, max_frames: int) -> list[float]:
    duration = max(0.0, segment.end - segment.start)
    if duration <= 1.0 or max_frames == 1:
        return [segment.start + duration / 2]

    ratios = [0.2, 0.5, 0.8]
    if max_frames >= 5:
        ratios = [0.12, 0.32, 0.5, 0.68, 0.88]
    return [segment.start + duration * r for r in ratios[:max_frames]]


def sample_times_by_audio_peaks(
    video: Path,
    segment: Segment,
    *,
    max_frames: int,
    window_seconds: float,
    min_spacing_seconds: float,
    sample_rate: int,
) -> list[float]:
    samples = read_segment_pcm(video, segment, sample_rate=sample_rate)
    if not samples:
        return fallback_sample_times(segment, max_frames)

    duration = max(0.0, segment.end - segment.start)
    if duration <= 1.0 or max_frames == 1:
        return [segment.start + duration / 2]

    window_size = max(1, int(sample_rate * window_seconds))
    hop_size = max(1, window_size // 2)
    scored: list[tuple[float, float]] = []
    for offset in range(0, max(1, len(samples) - window_size + 1), hop_size):
        window = samples[offset : offset + window_size]
        if not window:
            continue
        rms = sum(sample * sample for sample in window) / len(window)
        center = offset / sample_rate + len(window) / sample_rate / 2
        timestamp = min(segment.end - 0.05, max(segment.start + 0.05, segment.start + center))
        scored.append((rms, timestamp))

    selected: list[float] = []
    for _score, timestamp in sorted(scored, reverse=True):
        if all(abs(timestamp - existing) >= min_spacing_seconds for existing in selected):
            selected.append(timestamp)
        if len(selected) >= max_frames:
            break

    if not selected:
        return fallback_sample_times(segment, max_frames)
    return sorted(selected)


def sample_frames(args: argparse.Namespace) -> None:
    diarized = load_json(args.diarized)
    segments = parse_segments(diarized)
    args.frames_dir.mkdir(parents=True, exist_ok=True)

    log(
        "Sampling frames: "
        f"segments={len(segments)} max_frames_per_segment={args.max_frames} "
        f"video={args.video} frames_dir={args.frames_dir}"
    )
    manifest = []
    total_frames = 0
    for segment_index, segment in enumerate(segments, start=1):
        frame_paths = []
        try:
            timestamps = sample_times_by_audio_peaks(
                args.video,
                segment,
                max_frames=args.max_frames,
                window_seconds=args.audio_window_seconds,
                min_spacing_seconds=args.min_frame_spacing_seconds,
                sample_rate=args.audio_sample_rate,
            )
            sampling_method = "audio_peaks"
        except RuntimeError as exc:
            log(f"{exc}; falling back to ratio-based frame sampling for {segment.id}.")
            timestamps = fallback_sample_times(segment, args.max_frames)
            sampling_method = "fallback_ratios"
        log(
            f"Segment {segment_index}/{len(segments)} {segment.id}: "
            f"{segment.start:.3f}-{segment.end:.3f}s frames={len(timestamps)} method={sampling_method} "
            f"timestamps={', '.join(f'{t:.3f}' for t in timestamps)}"
        )
        for index, timestamp in enumerate(timestamps, start=1):
            frame_path = args.frames_dir / f"{segment.id}_{index}.jpg"
            run_ffmpeg([
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(args.video),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame_path),
            ])
            frame_paths.append(str(frame_path))
            total_frames += 1

        manifest.append(
            {
                "segment_id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "audio_speaker": segment.audio_speaker,
                "frame_sampling_method": sampling_method,
                "frame_timestamps": timestamps,
                "frames": frame_paths,
            }
        )

    save_json(args.manifest, manifest)
    log(f"Frame sampling complete: manifest={args.manifest} frames={total_frames}")


def transcribe_openai_diarized(
    *,
    config: GatewayConfig,
    audio: Path,
    model: str,
    language: str,
    chunking_strategy: str,
    known_speaker: list[str] | None = None,
) -> dict[str, Any]:
    base_fields: dict[str, str] = {
        "model": model,
        "response_format": "diarized_json",
        "language": language,
    }

    if known_speaker:
        names = []
        references = []
        for item in known_speaker:
            if "=" not in item:
                raise SystemExit("--known-speaker must use NAME=path format.")
            name, path = item.split("=", 1)
            names.append(name)
            references.append(image_or_audio_to_data_url(Path(path)))
        base_fields["known_speaker_names"] = json.dumps(names, ensure_ascii=False)
        base_fields["known_speaker_references"] = json.dumps(references, ensure_ascii=False)

    endpoints = [config.transcription_endpoint]
    if config.transcription_endpoint != "/audio/transcriptions":
        endpoints.append("/audio/transcriptions")

    chunking_variants = build_chunking_strategy_variants(chunking_strategy)
    last_error: Exception | None = None

    for endpoint in endpoints:
        for fields in apply_chunking_strategy_variants(base_fields, chunking_variants):
            body, content_type = build_multipart_form(fields, {"file": audio})
            req = request.Request(
                f"{config.base_url}{endpoint}",
                data=body,
                headers=gateway_headers(config, content_type),
                method="POST",
            )
            try:
                return gateway_request(config, req)
            except RuntimeError as exc:
                last_error = exc
                if not is_chunking_strategy_error(exc):
                    raise
                print(
                    "Compass rejected chunking_strategy="
                    f"{fields.get('chunking_strategy')} at endpoint={endpoint}; trying next variant.",
                    file=sys.stderr,
                )

    assert last_error is not None
    raise last_error


def build_chunking_strategy_variants(chunking_strategy: str) -> list[str | dict[str, str]]:
    text = (chunking_strategy or "auto").strip()
    if not text:
        text = "auto"
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return [parsed]
    if text != "auto":
        return [text]
    return [
        "auto",
        {"type": "server_vad"},
        {"type": "server_vad", "prefix_padding_ms": 300, "silence_duration_ms": 500},
    ]


def apply_chunking_strategy_variants(
    base_fields: dict[str, str],
    variants: list[str | dict[str, Any]],
) -> list[dict[str, str]]:
    applied: list[dict[str, str]] = []
    for variant in variants:
        fields = dict(base_fields)
        if isinstance(variant, str):
            fields["chunking_strategy"] = variant
        else:
            fields["chunking_strategy"] = json.dumps(variant, ensure_ascii=False)
        applied.append(fields)
    return applied


def transcribe(args: argparse.Namespace) -> None:
    config = load_gateway_config()
    payload = transcribe_openai_diarized(
        config=config,
        audio=args.audio,
        model=args.model or config.transcription_model,
        language=args.language,
        chunking_strategy=args.chunking_strategy,
        known_speaker=args.known_speaker,
    )
    save_json(args.out, payload)


def image_or_audio_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def extract_response_text(response_payload: dict[str, Any]) -> str:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in response_payload.get("output", []):
        if item.get("type") not in {None, "message"}:
            continue
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text)
    return "\n".join(chunks).strip()


def extract_gemini_text(response_payload: dict[str, Any]) -> str:
    text = response_payload.get("text")
    if isinstance(text, str) and text.strip():
        return text

    chunks: list[str] = []
    for candidate in response_payload.get("candidates", []):
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text)
    return repair_mojibake("\n".join(chunks).strip())


def repair_mojibake(text: str) -> str:
    if not text:
        return text
    markers = ("Ã", "Ä", "Æ", "áº", "á»", "Â")
    if not any(marker in text for marker in markers):
        return text
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text
    return repaired if score_mojibake(repaired) < score_mojibake(text) else text


def score_mojibake(text: str) -> int:
    markers = ("Ã", "Ä", "Æ", "áº", "á»", "Â", "€", "™")
    return sum(text.count(marker) for marker in markers)


def gemini_generate_content(
    *,
    config: GatewayConfig,
    model: str,
    prompt: str,
    media_path: Path | None,
    media_url: str | None,
    mime_type: str,
    audio_timestamp: bool = True,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    if media_url:
        parts.append({"fileData": {"fileUri": media_url, "mimeType": mime_type}})
    elif media_path:
        encoded = base64.b64encode(media_path.read_bytes()).decode("ascii")
        parts.append({"inlineData": {"mimeType": mime_type, "data": encoded}})
    else:
        raise SystemExit("Gemini call requires either media_path or media_url.")

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"audioTimestamp": audio_timestamp},
    }
    return gateway_request_json(config, f"/models/{model}:generateContent", payload, include_provider=False)


def render_openai_diarized_markdown(payload: dict[str, Any]) -> str:
    segments = parse_segments(payload)
    if not segments:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    lines = []
    for segment in segments:
        speaker = segment.audio_speaker or "Unknown"
        lines.append(f"[{format_time(segment.start)} - {format_time(segment.end)}] {speaker}: {segment.text}")
    return "\n".join(lines)


def bakeoff(args: argparse.Namespace) -> None:
    config = load_gateway_config()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sample_video = args.out_dir / "sample_clip.mp4"
    sample_audio = args.out_dir / "sample_audio.mp3"
    trim_video(args.video, sample_video, start=args.start, duration=args.duration)
    extract_transcription_audio(sample_video, sample_audio)

    outputs: dict[str, dict[str, Any]] = {}

    if not args.skip_openai:
        try:
            openai_payload = transcribe_openai_diarized(
                config=config,
                audio=sample_audio,
                model=args.openai_model or config.transcription_model,
                language=args.language,
                chunking_strategy=args.chunking_strategy,
                known_speaker=args.known_speaker,
            )
            outputs["openai_diarize"] = {
                "status": "ok",
                "model": args.openai_model or config.transcription_model,
                "endpoint": config.transcription_endpoint,
                "raw_path": str(args.out_dir / "openai_diarize.json"),
                "text": render_openai_diarized_markdown(openai_payload),
            }
            save_json(args.out_dir / "openai_diarize.json", openai_payload)
        except Exception as exc:
            error_payload = {
                "status": "error",
                "model": args.openai_model or config.transcription_model,
                "endpoint": config.transcription_endpoint,
                "error": str(exc),
            }
            outputs["openai_diarize"] = {
                **error_payload,
                "raw_path": str(args.out_dir / "openai_diarize_error.json"),
                "text": f"[OpenAI diarize failed]\n{exc}",
            }
            save_json(args.out_dir / "openai_diarize_error.json", error_payload)
            print(f"OpenAI diarize failed; continuing bake-off: {exc}", file=sys.stderr)

    if not args.skip_gemini_audio:
        gemini_audio_payload = gemini_generate_content(
            config=config,
            model=args.gemini_model or config.gemini_model,
            prompt=args.gemini_prompt,
            media_path=None if args.gemini_audio_url else sample_audio,
            media_url=args.gemini_audio_url,
            mime_type=args.gemini_audio_mime,
        )
        outputs["gemini_audio"] = {
            "status": "ok",
            "model": args.gemini_model or config.gemini_model,
            "endpoint": f"/models/{args.gemini_model or config.gemini_model}:generateContent",
            "raw_path": str(args.out_dir / "gemini_audio.json"),
            "text": extract_gemini_text(gemini_audio_payload),
        }
        save_json(args.out_dir / "gemini_audio.json", gemini_audio_payload)

    if args.include_gemini_video:
        gemini_video_payload = gemini_generate_content(
            config=config,
            model=args.gemini_model or config.gemini_model,
            prompt=args.gemini_prompt,
            media_path=None if args.gemini_video_url else sample_video,
            media_url=args.gemini_video_url,
            mime_type=args.gemini_video_mime,
        )
        outputs["gemini_video"] = {
            "status": "ok",
            "model": args.gemini_model or config.gemini_model,
            "endpoint": f"/models/{args.gemini_model or config.gemini_model}:generateContent",
            "raw_path": str(args.out_dir / "gemini_video.json"),
            "text": extract_gemini_text(gemini_video_payload),
        }
        save_json(args.out_dir / "gemini_video.json", gemini_video_payload)

    save_json(args.out_dir / "bakeoff_summary.json", outputs)
    write_bakeoff_markdown(args.out_dir / "bakeoff_comparison.md", args, sample_video, sample_audio, outputs)


def render_bakeoff(args: argparse.Namespace) -> None:
    outputs: dict[str, dict[str, Any]] = {}

    openai_path = args.out_dir / "openai_diarize.json"
    openai_error_path = args.out_dir / "openai_diarize_error.json"
    if openai_path.exists():
        payload = load_json(openai_path)
        outputs["openai_diarize"] = {
            "status": "ok",
            "model": payload.get("model", "gpt-4o-transcribe-diarize"),
            "endpoint": "/audio/internal/transcriptions",
            "raw_path": str(openai_path),
            "text": render_openai_diarized_markdown(payload),
        }
    elif openai_error_path.exists():
        payload = load_json(openai_error_path)
        outputs["openai_diarize"] = {
            "status": "error",
            "model": payload.get("model", "gpt-4o-transcribe-diarize"),
            "endpoint": payload.get("endpoint", "/audio/internal/transcriptions"),
            "raw_path": str(openai_error_path),
            "text": f"[OpenAI diarize failed]\n{payload.get('error', 'Unknown error')}",
        }

    gemini_audio_path = args.out_dir / "gemini_audio.json"
    if gemini_audio_path.exists():
        payload = load_json(gemini_audio_path)
        model = payload.get("modelVersion", "gemini")
        outputs["gemini_audio"] = {
            "status": "ok",
            "model": model,
            "endpoint": f"/models/{model}:generateContent",
            "raw_path": str(gemini_audio_path),
            "text": extract_gemini_text(payload),
        }

    gemini_video_path = args.out_dir / "gemini_video.json"
    if gemini_video_path.exists():
        payload = load_json(gemini_video_path)
        model = payload.get("modelVersion", "gemini")
        outputs["gemini_video"] = {
            "status": "ok",
            "model": model,
            "endpoint": f"/models/{model}:generateContent",
            "raw_path": str(gemini_video_path),
            "text": extract_gemini_text(payload),
        }

    if not outputs:
        raise SystemExit(f"No bake-off JSON files found in {args.out_dir}.")

    save_json(args.out_dir / "bakeoff_summary.json", outputs)
    sample_video = args.out_dir / "sample_clip.mp4"
    sample_audio = args.out_dir / "sample_audio.mp3"
    write_bakeoff_markdown(args.out_dir / "bakeoff_comparison.md", args, sample_video, sample_audio, outputs)


def write_bakeoff_markdown(
    out: Path,
    args: argparse.Namespace,
    sample_video: Path,
    sample_audio: Path,
    outputs: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# FGD Bake-off Comparison",
        "",
        f"Source video: `{getattr(args, 'video', '[rendered from existing outputs]')}`",
        f"Sample video: `{sample_video}`",
        f"Sample audio: `{sample_audio}`",
        f"Start: `{getattr(args, 'start', '[unknown]')}` seconds",
        f"Duration: `{getattr(args, 'duration', '[unknown]')}` seconds",
        "",
        "## Manual Scoring Rubric",
        "",
        "- Speaker count accuracy: did the model detect the right number of active speakers?",
        "- Speaker label consistency: does one label stay attached to one person?",
        "- Timestamp usefulness: are turn boundaries usable for review?",
        "- Overlap handling: are crosstalk sections flagged instead of invented?",
        "- Transcript readability: is the text good enough for cleanup?",
        "",
    ]

    for name, item in outputs.items():
        lines.extend(
            [
                f"## {name}",
                "",
                f"Model: `{item['model']}`",
                f"Endpoint: `{item['endpoint']}`",
                f"Raw JSON: `{item['raw_path']}`",
                "",
                "```text",
                item.get("text") or "[No text output parsed]",
                "```",
                "",
            ]
        )

    out.write_text("\n".join(lines), encoding="utf-8")


def assign_speakers(args: argparse.Namespace) -> None:
    participants = load_json(args.participants)
    manifest = load_json(args.manifest)
    manifest = select_manifest_items(manifest, start_segment=args.start_segment, limit=args.limit)
    model_name = args.model or os.environ.get("OPENAI_VISION_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")

    existing_results = load_existing_results(args.out) if args.resume else []
    existing_ids = {item.get("segment_id") for item in existing_results}
    pending_manifest = [item for item in manifest if item.get("segment_id") not in existing_ids]

    estimate = estimate_assign_speakers_cost(
        pending_manifest,
        model=model_name,
        image_input_tokens=args.est_image_input_tokens,
        text_input_tokens_per_segment=args.est_text_input_tokens_per_segment,
        output_tokens_per_segment=args.est_output_tokens_per_segment,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )
    print_assign_speakers_estimate(estimate, args.out, resumed=len(existing_results))
    if args.estimate_only:
        return
    if pending_manifest and not args.yes:
        answer = input("Run vision speaker assignment with this estimate? Type YES to continue: ").strip()
        if answer != "YES":
            raise SystemExit("Cancelled before model calls.")

    results = list(existing_results)
    if not pending_manifest:
        log(f"No pending segments. Existing output already has {len(existing_results)} results: {args.out}")
        return

    config = load_gateway_config()
    model_name = args.model or config.vision_model
    log(
        "Starting vision speaker assignment: "
        f"pending_segments={len(pending_manifest)} existing_results={len(existing_results)} "
        f"model={model_name}"
    )

    for item_index, item in enumerate(pending_manifest, start=1):
        segment_id = item.get("segment_id", f"pending_{item_index}")
        frames = item.get("frames", [])
        log(
            f"Vision segment {item_index}/{len(pending_manifest)} {segment_id}: "
            f"{item.get('start')}-{item.get('end')}s frames={len(frames)}"
        )
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Identify the most likely visible speaker for this FGD segment. "
                    "Use the participant map, frame evidence, and transcript text. "
                    "Return only JSON with keys: segment_id, visible_speaker_id, "
                    "confidence, evidence, flags. Use visible_speaker_id null when unclear.\n\n"
                    f"Participant map:\n{json.dumps(participants, ensure_ascii=False)}\n\n"
                    f"Segment:\n{json.dumps(item, ensure_ascii=False)}"
                ),
            }
        ]
        for frame in frames:
            content.append({"type": "input_image", "image_url": image_or_audio_to_data_url(Path(frame))})

        payload = {
            "model": model_name,
            "input": [{"role": "user", "content": content}],
            "temperature": 0,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "fgd_visible_speaker_assignment",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "segment_id": {"type": "string"},
                            "visible_speaker_id": {"type": ["string", "null"]},
                            "confidence": {"type": "number"},
                            "evidence": {"type": "string"},
                            "flags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["segment_id", "visible_speaker_id", "confidence", "evidence", "flags"],
                        "additionalProperties": False,
                    },
                }
            },
        }
        raw = gateway_request_json(config, "/responses", payload)
        text = extract_response_text(raw)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {
                "segment_id": item["segment_id"],
                "visible_speaker_id": None,
                "confidence": 0,
                "evidence": text,
                "flags": ["invalid_json_from_model"],
            }
        results.append(parsed)
        save_json(args.out, results)
        log(
            f"Saved {len(results)} total speaker assignments to {args.out}; "
            f"latest={segment_id} visible_speaker_id={parsed.get('visible_speaker_id')} "
            f"confidence={parsed.get('confidence')}"
        )

    log(f"Vision speaker assignment complete: {args.out}")


def select_manifest_items(
    manifest: list[dict[str, Any]],
    *,
    start_segment: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    items = manifest
    if start_segment:
        start_index = next(
            (i for i, item in enumerate(items) if item.get("segment_id") == start_segment),
            None,
        )
        if start_index is None:
            raise SystemExit(f"--start-segment {start_segment} was not found in manifest.")
        items = items[start_index:]
    if limit is not None:
        items = items[: max(0, limit)]
    return items


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, list):
        raise SystemExit(f"Existing output is not a JSON list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def estimate_assign_speakers_cost(
    manifest: list[dict[str, Any]],
    *,
    model: str,
    image_input_tokens: int,
    text_input_tokens_per_segment: int,
    output_tokens_per_segment: int,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> dict[str, Any]:
    segment_count = len(manifest)
    frame_count = sum(len(item.get("frames", [])) for item in manifest)
    input_tokens = frame_count * image_input_tokens + segment_count * text_input_tokens_per_segment
    output_tokens = segment_count * output_tokens_per_segment
    input_cost = input_tokens / 1_000_000 * input_cost_per_1m
    output_cost = output_tokens / 1_000_000 * output_cost_per_1m
    return {
        "model": model,
        "segments": segment_count,
        "frames": frame_count,
        "image_input_tokens_per_frame": image_input_tokens,
        "text_input_tokens_per_segment": text_input_tokens_per_segment,
        "output_tokens_per_segment": output_tokens_per_segment,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "input_cost_per_1m": input_cost_per_1m,
        "output_cost_per_1m": output_cost_per_1m,
        "estimated_input_cost": input_cost,
        "estimated_output_cost": output_cost,
        "estimated_total_cost": input_cost + output_cost,
    }


def print_assign_speakers_estimate(estimate: dict[str, Any], out: Path, *, resumed: int) -> None:
    lines = [
        "Vision speaker assignment estimate",
        f"- model: {estimate['model']}",
        f"- pending segments: {estimate['segments']}",
        f"- pending frames: {estimate['frames']}",
        f"- already completed segments: {resumed}",
        f"- output path: {out}",
        f"- estimated input tokens: {estimate['estimated_input_tokens']:,}",
        f"- estimated output tokens: {estimate['estimated_output_tokens']:,}",
        f"- input price used: ${estimate['input_cost_per_1m']:.4f} / 1M tokens",
        f"- output price used: ${estimate['output_cost_per_1m']:.4f} / 1M tokens",
        f"- estimated total: ${estimate['estimated_total_cost']:.4f}",
    ]
    print("\n".join(lines), file=sys.stderr, flush=True)


def detect_overlap(segment: Segment, previous: Segment | None) -> bool:
    return bool(previous and segment.start < previous.end)


def fuse(args: argparse.Namespace) -> None:
    diarized = load_json(args.diarized)
    participants = {p["id"]: p for p in load_json(args.participants)}
    vision_rows = {v["segment_id"]: v for v in load_json(args.vision)}
    segments = parse_segments(diarized)

    lines = ["# FGD Transcript", ""]
    previous: Segment | None = None
    review_count = 0

    for segment in segments:
        vision = vision_rows.get(segment.id, {})
        visible_id = vision.get("visible_speaker_id")
        confidence = float(vision.get("confidence") or 0)
        flags = set(vision.get("flags") or [])

        if confidence < args.min_confidence:
            flags.add("low_visual_confidence")
        if not visible_id:
            flags.add("no_visual_assignment")
        if segment.end - segment.start < args.short_segment_seconds:
            flags.add("short_segment")
        if detect_overlap(segment, previous):
            flags.add("possible_overlap")

        speaker_name = participants.get(visible_id, {}).get("display_name") or visible_id or segment.audio_speaker or "Unknown"
        timestamp = f"{format_time(segment.start)}-{format_time(segment.end)}"
        flag_text = ""
        if flags:
            review_count += 1
            flag_text = " " + " ".join(f"`{flag}`" for flag in sorted(flags))

        lines.append(f"## {timestamp} {speaker_name}{flag_text}")
        lines.append("")
        lines.append(segment.text or "[No transcript text]")
        if vision.get("evidence"):
            lines.append("")
            lines.append(f"_Vision evidence: {vision['evidence']}_")
        lines.append("")
        previous = segment

    lines.insert(2, f"Review-flagged segments: {review_count}")
    lines.insert(3, "")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")


def format_time(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_parser() -> argparse.ArgumentParser:
    load_dotenv()
    parser = argparse.ArgumentParser(description="FGD audio + vision transcription helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract-audio")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.set_defaults(func=extract_audio)

    p = sub.add_parser("transcribe")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-transcribe-diarize"))
    p.add_argument("--language", default="th")
    p.add_argument("--chunking-strategy", default="auto")
    p.add_argument(
        "--known-speaker",
        action="append",
        default=[],
        help="Optional NAME=audio_path reference clip. Repeat for up to the API-supported limit.",
    )
    p.set_defaults(func=transcribe)

    p = sub.add_parser("bakeoff")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs") / "bakeoff")
    p.add_argument("--start", type=float, default=0.0, help="Sample start time in seconds.")
    p.add_argument("--duration", type=float, default=600.0, help="Sample duration in seconds.")
    p.add_argument("--language", default="th")
    p.add_argument("--chunking-strategy", default="auto")
    p.add_argument("--openai-model", default=os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-transcribe-diarize"))
    p.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    p.add_argument(
        "--gemini-prompt",
        default=(
            "Transcribe this focus group discussion in the format "
            "[HH:MM:SS.mmm - HH:MM:SS.mmm] Speaker A: caption. "
            "Each line must be one continuous speaking turn with exact begin and end timestamps. "
            "Use distinct speaker labels and keep labels consistent. "
            "Flag overlapping or unclear speech as [overlap] or [unclear] instead of inventing words."
        ),
    )
    p.add_argument("--gemini-audio-url", default=None, help="Optional Compass-accessible audio URL for Gemini fileData.")
    p.add_argument("--gemini-video-url", default=None, help="Optional Compass-accessible video URL for Gemini fileData.")
    p.add_argument("--gemini-audio-mime", default="audio/mpeg")
    p.add_argument("--gemini-video-mime", default="video/mp4")
    p.add_argument("--include-gemini-video", action="store_true")
    p.add_argument("--skip-openai", action="store_true")
    p.add_argument("--skip-gemini-audio", action="store_true")
    p.add_argument(
        "--known-speaker",
        action="append",
        default=[],
        help="Optional NAME=audio_path reference clip for OpenAI diarization.",
    )
    p.set_defaults(func=bakeoff)

    p = sub.add_parser("render-bakeoff")
    p.add_argument("--out-dir", type=Path, default=Path("outputs") / "bakeoff")
    p.set_defaults(func=render_bakeoff)

    p = sub.add_parser("segmentize-gemini")
    p.add_argument("--gemini-json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--clip-duration", type=float, default=None)
    p.add_argument("--default-tail-seconds", type=float, default=2.0)
    p.set_defaults(func=segmentize_gemini)

    p = sub.add_parser("sample-frames")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--diarized", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=3)
    p.add_argument("--audio-window-seconds", type=float, default=0.35)
    p.add_argument("--min-frame-spacing-seconds", type=float, default=0.55)
    p.add_argument("--audio-sample-rate", type=int, default=16000)
    p.set_defaults(func=sample_frames)

    p = sub.add_parser("assign-speakers")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--participants", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=os.environ.get("OPENAI_VISION_MODEL"))
    p.add_argument("--estimate-only", action="store_true", help="Print estimate and exit before model calls.")
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt after printing the estimate.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N pending manifest segments.")
    p.add_argument("--start-segment", default=None, help="Start processing at this segment_id.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--est-image-input-tokens",
        type=int,
        default=int(os.environ.get("OPENAI_VISION_TOKENS_PER_IMAGE", "765")),
        help="Rough token estimate per image frame.",
    )
    p.add_argument(
        "--est-text-input-tokens-per-segment",
        type=int,
        default=int(os.environ.get("OPENAI_VISION_TEXT_TOKENS_PER_SEGMENT", "350")),
    )
    p.add_argument(
        "--est-output-tokens-per-segment",
        type=int,
        default=int(os.environ.get("OPENAI_VISION_OUTPUT_TOKENS_PER_SEGMENT", "140")),
    )
    p.add_argument(
        "--input-cost-per-1m",
        type=float,
        default=float(os.environ.get("OPENAI_VISION_INPUT_COST_PER_1M", "2.50")),
    )
    p.add_argument(
        "--output-cost-per-1m",
        type=float,
        default=float(os.environ.get("OPENAI_VISION_OUTPUT_COST_PER_1M", "10.00")),
    )
    p.set_defaults(func=assign_speakers)

    p = sub.add_parser("fuse")
    p.add_argument("--diarized", type=Path, required=True)
    p.add_argument("--vision", type=Path, required=True)
    p.add_argument("--participants", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-confidence", type=float, default=0.7)
    p.add_argument("--short-segment-seconds", type=float, default=1.0)
    p.set_defaults(func=fuse)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
