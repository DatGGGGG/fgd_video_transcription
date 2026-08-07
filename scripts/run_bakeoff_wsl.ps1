param(
    [string]$Distro = "Ubuntu-D",
    [string]$Video = "inputs\fgd_sample_hbs_5_min_clean.mp4",
    [string]$OutDir = "outputs\bakeoff",
    [double]$Start = 0,
    [double]$Duration = 600,
    [string]$GeminiModel = "",
    [switch]$IncludeGeminiVideo,
    [switch]$SkipOpenAI,
    [switch]$SkipGeminiAudio
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath {
    param([string]$PathValue)
    $resolved = Resolve-Path -LiteralPath $PathValue
    $drive = $resolved.Path.Substring(0, 1).ToLowerInvariant()
    $rest = $resolved.Path.Substring(2).Replace("\", "/")
    return "/mnt/$drive$rest"
}

function Convert-DirToWslPath {
    param([string]$PathValue)
    if (-not (Test-Path -LiteralPath $PathValue)) {
        New-Item -ItemType Directory -Force -Path $PathValue | Out-Null
    }
    return Convert-ToWslPath $PathValue
}

function Convert-ToBashSingleQuoted {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

$repoWin = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$repoDrive = $repoWin.Path.Substring(0, 1).ToLowerInvariant()
$repoRest = $repoWin.Path.Substring(2).Replace("\", "/")
$repoWsl = "/mnt/$repoDrive$repoRest"

$videoWsl = Convert-ToWslPath $Video
$outDirWsl = Convert-DirToWslPath $OutDir

$envNames = @(
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_PROVIDER",
    "OPENAI_MODEL",
    "OPENAI_TRANSCRIPTION_MODEL",
    "OPENAI_TRANSCRIPTION_ENDPOINT",
    "OPENAI_VISION_MODEL",
    "GEMINI_MODEL",
    "OPENAI_TIMEOUT_SECONDS",
    "OPENAI_MAX_RETRIES"
)

$exports = @()
foreach ($name in $envNames) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ($null -ne $value -and $value.Trim()) {
        $exports += "export $name=$(Convert-ToBashSingleQuoted $value)"
    }
}

$pythonArgs = @(
    "python3 scripts/fgd_pipeline.py bakeoff",
    "--video '$videoWsl'",
    "--out-dir '$outDirWsl'",
    "--start $Start",
    "--duration $Duration"
)

if ($GeminiModel.Trim()) {
    $pythonArgs += "--gemini-model '$GeminiModel'"
}
if ($IncludeGeminiVideo) {
    $pythonArgs += "--include-gemini-video"
}
if ($SkipOpenAI) {
    $pythonArgs += "--skip-openai"
}
if ($SkipGeminiAudio) {
    $pythonArgs += "--skip-gemini-audio"
}

$steps = @("cd '$repoWsl'")
if ($exports.Count -gt 0) {
    $steps += ($exports -join "; ")
}
$steps += "if [ -d .venv ]; then . .venv/bin/activate; fi"
$steps += ($pythonArgs -join " ")

$command = $steps -join "; "
wsl -d $Distro -- bash -lc $command
