param(
    [int[]]$Seeds,
    [switch]$Force,
    [switch]$Smoke,
    [switch]$CleanCache
)
$ErrorActionPreference = 'Stop'

$pipelinePython = (Get-Command python).Source
if (-not (Test-Path -LiteralPath $pipelinePython)) {
    throw "Python nao encontrado: $pipelinePython"
}

$pipelineArgs = @(
    '-m', 'our_src.part3.run',
    '--config', (Join-Path $PSScriptRoot 'our_src/part3/config.yaml')
)
if ($Seeds.Count -gt 0) { $pipelineArgs += @('--seeds') + $Seeds }
if ($Force) { $pipelineArgs += '--force' }
if ($Smoke) { $pipelineArgs += '--smoke' }
if ($CleanCache) { $pipelineArgs += '--clean-cache' }

# 'python -m' only finds our_src when run from code/.
Push-Location -LiteralPath $PSScriptRoot
try {
    & $pipelinePython @pipelineArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
