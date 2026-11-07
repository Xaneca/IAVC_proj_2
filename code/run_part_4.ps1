param(
    [ValidateSet('A','B','C','D','E','F')][string[]]$Stages = @('A','B','C','D','E','F'),
    [switch]$DryRun,
    [switch]$Smoke,
    [int]$PerClass = 2,
    [switch]$Final,
    [string]$Output
)
$ErrorActionPreference = 'Stop'

$pipelinePython = (Get-Command python).Source
if (-not (Test-Path -LiteralPath $pipelinePython)) {
    throw "Python nao encontrado: $pipelinePython"
}

$pipelineArgs = @(
    '-m', 'our_src.part4.run',
    '--config', (Join-Path $PSScriptRoot 'our_src/part4/config.json'),
    '--stages'
) + $Stages
if ($DryRun) { $pipelineArgs += '--dry-run' }
if ($Smoke) { $pipelineArgs += @('--smoke', $PerClass) }
if ($Final) { $pipelineArgs += '--final' }
# Resolved here because the run below happens from code/, not the caller's folder.
if ($Output) { $pipelineArgs += @('--output', $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)) }

# 'python -m' only finds our_src when run from code/.
Push-Location -LiteralPath $PSScriptRoot
try {
    & $pipelinePython @pipelineArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
