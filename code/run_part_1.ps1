param(
    [ValidateSet('A','B','C','D','E','F','G')][string]$Start = 'A',
    [ValidateSet('A','B','C','D','E','F','G')][string]$End = 'G',
    [switch]$DryRun,
    [switch]$Smoke,
    [int]$PerClass = 2
)
$ErrorActionPreference = 'Stop'

$pipelinePython = (Get-Command python).Source
if (-not (Test-Path -LiteralPath $pipelinePython)) {
    throw "Python não encontrado: $pipelinePython"
}

$pipelineArgs = @(
    '-m', 'our_src.part1.run',
    '--project', $PSScriptRoot,
    '--python', $pipelinePython,
    '--start', $Start,
    '--end', $End
)
if ($DryRun) { $pipelineArgs += '--dry-run' }
if ($Smoke) { $pipelineArgs += @('--smoke', '--per-class', $PerClass) }

# 'python -m' only finds our_src when run from code/.
Push-Location -LiteralPath $PSScriptRoot
try {
    & $pipelinePython @pipelineArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
