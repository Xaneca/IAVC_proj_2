param(
    [ValidateSet('H','I')][string[]]$Stages = @('H','I'),
    [switch]$Smoke,
    [int]$PerClass = 2,
    [switch]$Preflight,
    [string]$SiftCache,
    [int]$Threads = 2
)
$ErrorActionPreference = 'Stop'

$pipelinePython = (Get-Command python).Source
if (-not (Test-Path -LiteralPath $pipelinePython)) {
    throw "Python nao encontrado: $pipelinePython"
}

$pipelineArgs = @(
    '-m', 'our_src.part2.run',
    '--stages'
) + $Stages + @('--threads', $Threads)
if ($Smoke) { $pipelineArgs += @('--smoke', '--per-class', $PerClass) }
if ($Preflight) { $pipelineArgs += '--preflight' }
if ($SiftCache) { $pipelineArgs += @('--sift-cache', $SiftCache) }

# 'python -m' only finds our_src when run from code/.
Push-Location -LiteralPath $PSScriptRoot
try {
    & $pipelinePython @pipelineArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
