param(
    [ValidateSet('A','B','C','D','E','F','G')][string]$Start = 'A',
    [ValidateSet('A','B','C','D','E','F','G')][string]$End = 'G',
    [switch]$DryRun,
    [switch]$Smoke,
    [int]$PerClass = 2
)
$ErrorActionPreference = 'Stop'
$pipelinePython = Join-Path $PSScriptRoot '.venv-stage-a\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pipelinePython)) { throw "Python não encontrado: $pipelinePython" }
$pipelineArgs = @((Join-Path $PSScriptRoot 'run_pipeline.py'), '--project', $PSScriptRoot, '--start', $Start, '--end', $End)
if ($DryRun) { $pipelineArgs += '--dry-run' }
if ($Smoke) { $pipelineArgs += @('--smoke', '--per-class', $PerClass) }
& $pipelinePython @pipelineArgs
exit $LASTEXITCODE
