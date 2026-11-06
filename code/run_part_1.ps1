param(
    [ValidateSet('A','B','C','D','E','F','G','H')][string]$Start = 'A',
    [ValidateSet('A','B','C','D','E','F','G','H')][string]$End = 'H',
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
    (Join-Path $PSScriptRoot 'run_part_1.py'),
    '--project', $PSScriptRoot,
    '--python', $pipelinePython,
    '--start', $Start,
    '--end', $End
) 
if ($DryRun) { $pipelineArgs += '--dry-run' }
if ($Smoke) { $pipelineArgs += @('--smoke', '--per-class', $PerClass) }
& $pipelinePython @pipelineArgs
exit $LASTEXITCODE
