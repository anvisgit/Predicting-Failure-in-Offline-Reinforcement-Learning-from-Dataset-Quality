param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ScratchPython = "C:\Users\Anvi\.gemini\antigravity\scratch\offline-rl-research\venv\Scripts\python.exe"
$ProjectPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (Test-Path $ScratchPython) {
    $Python = $ScratchPython
} elseif (Test-Path $ProjectPython) {
    $Python = $ProjectPython
} else {
    throw "No project Python found. Create a venv or update scripts\run.ps1 with your venv path."
}

Write-Host "Using Python: $Python"
& $Python (Join-Path $ProjectRoot "scripts\train.py") @Args
