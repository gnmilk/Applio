# Execute the four user-authorized local commits. No push, reset, stash or amend.
# Run only after reviewing commit-series.json and passing safe verification.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ($repoRoot -ne 'E:\AiVoice\Applio_3.6.5') {
    throw "Unexpected repository: $repoRoot"
}
Set-Location -LiteralPath $repoRoot
$gitExe = 'C:\Program Files\Git\cmd\git.exe'
$pythonExe = Join-Path $repoRoot 'env\python.exe'
$plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'commit-series.json') -Raw | ConvertFrom-Json

function Invoke-Git {
    param([string[]]$Arguments)
    $output = @(& $gitExe @Arguments)
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $($Arguments -join ' ')" }
    return $output
}

function Assert-Stage {
    param([string[]]$ExpectedFiles, [switch]$ToolsOnly)
    $actual = @(Invoke-Git -Arguments @('diff', '--cached', '--name-only'))
    if ($actual.Count -eq 0) { throw 'Nothing staged for this commit' }
    if ($ToolsOnly) {
        foreach ($file in $actual) {
            if (-not $file.StartsWith('tools/realtime_migration/')) {
                throw "Unexpected staged file: $file"
            }
        }
    } elseif (@(Compare-Object ($ExpectedFiles | Sort-Object) ($actual | Sort-Object)).Count -gt 0) {
        throw "Staged files differ from this commit's explicit file list"
    }
    # Original backups and upstream patches must retain their original bytes,
    # including any pre-existing upstream whitespace.
    Invoke-Git -Arguments @('diff', '--cached', '--check', '--', '.',
        ':(exclude)tools/realtime_migration/backup-20261010/**',
        ':(exclude)tools/realtime_migration/evidence/*.patch') | Out-Host
    Invoke-Git -Arguments @('diff', '--cached', '--stat') | Out-Host
}

if ((Invoke-Git -Arguments @('rev-parse', '--show-toplevel')).Replace('/', '\') -ne $repoRoot) {
    throw 'Repository root mismatch'
}
if ((Invoke-Git -Arguments @('branch', '--show-current')) -ne $plan.branch) {
    throw 'Branch changed since commit planning'
}
if ((Invoke-Git -Arguments @('rev-parse', 'HEAD')) -ne $plan.base) {
    throw 'HEAD changed since commit planning; inspect any partial commits before continuing'
}
if (@(Invoke-Git -Arguments @('diff', '--cached', '--name-only')).Count -ne 0) {
    throw 'Existing staged changes must be preserved; nothing will be committed'
}
$configHash = (Get-FileHash -LiteralPath 'assets/config.json' -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedHead = $plan.base
$testModules = @(
    @('tools.realtime_migration.test_volume'),
    @('tools.realtime_migration.test_audio_devices', 'tools.realtime_migration.test_lifecycle'),
    @('tools.realtime_migration.test_custom_embedder')
)
$bodies = @(
    'Keep the model output amplitude by removing the final extra input RMS multiplication. Preserve VAD, silence threshold, warmup, volume envelope and crossfade. Include CPU regression coverage.',
    'Resolve name and Host API identities again at stream creation, validate stream parameters, and serialize start/stop cleanup. Preserve bounded nonblocking callbacks and avoid automatic WDM-KS fallback after the observed Windows bugcheck. Include mocked device and lifecycle regressions; physical audio is not verified.',
    'Fail clearly when a requested realtime custom embedder directory is unavailable instead of silently falling back to ContentVec. Keep the default loader behavior for other callers and preserve upstream custom-path cache invalidation. Include loader and cache regressions.',
    'Preserve the three-way source audit, original code backups, migration decisions and safe verification tools. Record the interrupted hardware test and WDF_VIOLATION limitation without claiming realtime hardware success.'
)

for ($i = 0; $i -lt 4; $i++) {
    if ((Invoke-Git -Arguments @('rev-parse', 'HEAD')) -ne $expectedHead) {
        throw 'HEAD changed during execution'
    }
    if ((Get-FileHash -LiteralPath 'assets/config.json' -Algorithm SHA256).Hash.ToLowerInvariant() -ne $configHash) {
        throw 'Configuration changed during execution; preserve it and stop'
    }
    $entry = $plan.commits[$i]
    Write-Host "Preparing commit $($i + 1)/4: $($entry.title)"
    if ($i -lt 3) {
        $pythonArgs = @('-B', '-X', 'utf8', '-m', 'unittest', '-v') + @($testModules[$i])
        & $pythonExe @pythonArgs
        if ($LASTEXITCODE -ne 0) { throw "Regression tests failed for commit $($i + 1)" }
        Invoke-Git -Arguments (@('add', '--') + @($entry.files)) | Out-Host
        Assert-Stage -ExpectedFiles @($entry.files)
    } else {
        Invoke-Git -Arguments @('add', '--', 'tools/realtime_migration') | Out-Host
        & $pythonExe -B -X utf8 tools/realtime_migration/verify.py --config-sha256 $configHash
        if ($LASTEXITCODE -ne 0) { throw 'Full safe verification failed' }
        Invoke-Git -Arguments @('add', '--', 'tools/realtime_migration/evidence/final-verification.json', 'tools/realtime_migration/evidence/regression-tests.log') | Out-Host
        Assert-Stage -ToolsOnly
    }
    Invoke-Git -Arguments @('commit', '-m', $entry.title, '-m', $bodies[$i]) | Out-Host
    $expectedHead = Invoke-Git -Arguments @('rev-parse', 'HEAD')
}
Invoke-Git -Arguments @('log', '-4', '--oneline') | Out-Host
Invoke-Git -Arguments @('status', '--short') | Out-Host
if (@(Invoke-Git -Arguments @('status', '--porcelain')).Count -ne 0) {
    throw 'Commits completed, but working tree contains remaining changes; inspect them'
}
Write-Host 'Four local commits completed; working tree is clean. Nothing was pushed.'
