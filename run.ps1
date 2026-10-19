$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# Pick the Windows Python launcher when available; otherwise use python on PATH.
$launcher = Get-Command py -ErrorAction SilentlyContinue
if ($launcher) {
    $python = $launcher.Source
    $pythonArgs = $null
    foreach ($version in @("3.13", "3.12", "3.11", "3.10", "3.9")) {
        try {
            $versionOutput = & $python "-$version" --version 2>$null
        } catch {
            continue
        }
        if ($LASTEXITCODE -eq 0) {
            $pythonArgs = @("-$version")
            break
        }
    }
    if ($null -eq $pythonArgs) {
        throw "Python 3.9 through 3.13 is required. Install a supported Python version and try again."
    }
} else {
    $python = (Get-Command python -ErrorAction Stop).Source
    $versionOutput = & $python --version
    if ($LASTEXITCODE -ne 0 -or $versionOutput -notmatch "Python 3\.(9|10|11|12|13)(\.|$)") {
        throw "Python 3.9 through 3.13 is required. Install a supported Python version and try again."
    }
    $pythonArgs = @()
}

& $python @pythonArgs --version
if ($LASTEXITCODE -ne 0) { throw "Python could not be started." }

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $python @pythonArgs -m venv (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }

& $venvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install project dependencies." }

$modelPath = Join-Path $PSScriptRoot "models\yolov8n.pt"
if (-not (Test-Path -LiteralPath $modelPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $modelPath) | Out-Null
    $modelUrl = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
}

& $venvPython (Join-Path $PSScriptRoot "main.py") @args
exit $LASTEXITCODE
