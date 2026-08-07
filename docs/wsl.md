# WSL Runbook

The intended WSL distro name is `Ubuntu-D`.

## 1. Confirm WSL

From PowerShell:

```powershell
wsl -l -v
```

If `Ubuntu-D` is missing, install or import that distro first. This machine currently reports no installed WSL distributions, so repo scripts are ready but cannot run until WSL has a distro.

## 2. Bootstrap Ubuntu

Once `Ubuntu-D` exists:

```powershell
wsl -d Ubuntu-D -- bash -lc "cd /mnt/d/Coding/fgd_video_transcription && bash scripts/bootstrap_ubuntu.sh"
```

This installs:

- `python3`
- `python3-venv`
- `python3-pip`
- `ffmpeg`

It also creates `.venv` in the repo.

## 3. Configure Compass

Inside Ubuntu, create a persistent `.env`:

```bash
cd /mnt/d/Coding/fgd_video_transcription
cp .env.example .env
nano .env
```

The CLI loads `.env` automatically. Manual exports also work:

```bash
export OPENAI_API_KEY="replace_me"
export OPENAI_BASE_URL="https://compass.llm.shopee.io/compass-api/v1"
export OPENAI_PROVIDER="OpenAI"
export OPENAI_TRANSCRIPTION_MODEL="gpt-4o-transcribe-diarize"
export OPENAI_TRANSCRIPTION_ENDPOINT="/audio/internal/transcriptions"
export GEMINI_MODEL="gemini-2.5-flash"
```

## 4. Run Bake-off

PowerShell wrapper:

```powershell
.\scripts\run_bakeoff_wsl.ps1 -Distro Ubuntu-D -Video inputs\fgd_sample_hbs_5_min_clean.mp4 -OutDir outputs\bakeoff -Duration 600
```

Include Gemini video:

```powershell
.\scripts\run_bakeoff_wsl.ps1 -Distro Ubuntu-D -Video inputs\fgd_sample_hbs_5_min_clean.mp4 -OutDir outputs\bakeoff -Duration 600 -IncludeGeminiVideo
```

Direct Ubuntu command:

```bash
cd /mnt/d/Coding/fgd_video_transcription
source .venv/bin/activate
python3 scripts/fgd_pipeline.py bakeoff --video inputs/fgd_sample_hbs_5_min_clean.mp4 --out-dir outputs/bakeoff --start 0 --duration 600
```
