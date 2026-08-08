$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$harnessEnv = conda env list --json | ConvertFrom-Json | Select-Object -ExpandProperty envs | Where-Object { $_ -match '[\\/]envs[\\/]llm-harness$' } | Select-Object -First 1
if (-not $harnessEnv) {
    throw 'Conda environment llm-harness was not found. Run scripts/bootstrap.ps1 first.'
}
$env:PATH = "$harnessEnv;$env:PATH"
Push-Location "$repoRoot\frontend"
try {
    & "$harnessEnv\npm.cmd" run build
} finally {
    Pop-Location
}
Set-Location $repoRoot
conda run --no-capture-output -n llm-harness python -m harness
