$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$envExists = conda env list --json | ConvertFrom-Json | Select-Object -ExpandProperty envs | Where-Object { $_ -match '[\\/]envs[\\/]llm-harness$' }
if (-not $envExists) {
    conda create -n llm-harness python=3.11 nodejs=20 -y
}
conda run -n llm-harness python -m pip install -e "$repoRoot[dev]"
$harnessEnv = $envExists | Select-Object -First 1
if (-not $harnessEnv) {
    $harnessEnv = conda env list --json | ConvertFrom-Json | Select-Object -ExpandProperty envs | Where-Object { $_ -match '[\\/]envs[\\/]llm-harness$' } | Select-Object -First 1
}
$env:PATH = "$harnessEnv;$env:PATH"
Push-Location "$repoRoot\frontend"
try {
    & "$harnessEnv\npm.cmd" install
} finally {
    Pop-Location
}
Write-Host 'Bootstrap complete. Run scripts/dev.ps1.'
