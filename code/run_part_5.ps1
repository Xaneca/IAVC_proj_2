param(
    [string[]]$Only,
    [switch]$Force,
    [switch]$Smoke,
    [switch]$FiguresOnly
)
$ErrorActionPreference = 'Stop'

$pipelinePython = (Get-Command python).Source
if (-not (Test-Path -LiteralPath $pipelinePython)) {
    throw "Python nao encontrado: $pipelinePython"
}

$pipelineArgs = @(
    '-m', 'our_src.part5.run',
    '--config', (Join-Path $PSScriptRoot 'our_src/part5/config.yaml')
)
if ($Only.Count -gt 0) { $pipelineArgs += @('--only') + $Only }
if ($Force) { $pipelineArgs += '--force' }
if ($Smoke) { $pipelineArgs += '--smoke' }
if ($FiguresOnly) { $pipelineArgs += '--figures-only' }

# 'python -m' only finds our_src when run from code/.
Push-Location -LiteralPath $PSScriptRoot
try {
    & $pipelinePython @pipelineArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
