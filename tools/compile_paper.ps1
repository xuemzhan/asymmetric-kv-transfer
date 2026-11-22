# Build every edition of the paper with the repository-local Tectonic binary.
#
#   English edition      paper/main.pdf
#   Chinese edition      paper/main_zh.pdf
#   arXiv bundle         paper/arxiv_submission/main.pdf
#
# The arXiv bundle is refreshed from paper/main.tex first, so the two English
# sources stay byte-identical (the SYNC assertion in
# paper/audit/verify_audit4_edits.py checks exactly that).
#
# Usage (from the repository root):
#   powershell -File tools/compile_paper.ps1
#   powershell -File tools/compile_paper.ps1 -SkipArxiv
#
# The compiled PDFs are gitignored on purpose: the sources are the deliverable.

param(
    [switch]$SkipArxiv
)

$ErrorActionPreference = 'Stop'

$ToolsDir = $PSScriptRoot
$RepoDir = Split-Path -Parent $ToolsDir
$Tectonic = Join-Path $ToolsDir 'tectonic\tectonic.exe'
$PaperDir = Join-Path $RepoDir 'paper'
$ArxivDir = Join-Path $PaperDir 'arxiv_submission'

if (-not (Test-Path -LiteralPath $Tectonic)) {
    throw "Tectonic binary not found: $Tectonic"
}

function Invoke-Tectonic {
    param([string]$WorkDir, [string]$Source)

    Push-Location $WorkDir
    try {
        & $Tectonic --keep-logs $Source
        if ($LASTEXITCODE -ne 0) {
            throw "tectonic failed on $Source in $WorkDir (exit $LASTEXITCODE)"
        }
    }
    finally {
        Pop-Location
    }
}

function Show-Build {
    param([string]$WorkDir, [string]$Stem)

    $log = Join-Path $WorkDir ($Stem + '.log')
    if (-not (Test-Path -LiteralPath $log)) {
        Write-Host "  $Stem : no log written" -ForegroundColor Yellow
        return
    }
    $text = Get-Content -LiteralPath $log -Raw
    $pages = if ($text -match 'Output written on .*?\((\d+) pages') { $Matches[1] } else { '?' }
    $undefined = ([regex]::Matches($text, 'undefined|Undefined')).Count
    $overfull = ([regex]::Matches($text, 'Overfull \\hbox')).Count
    $warnings = ([regex]::Matches($text, 'LaTeX Warning')).Count
    $missing = ([regex]::Matches($text, 'Missing character')).Count
    $status = if (($undefined + $overfull + $warnings + $missing) -eq 0) { 'clean' } else { 'CHECK' }
    $colour = if ($status -eq 'clean') { 'Green' } else { 'Yellow' }
    Write-Host ("  {0,-9} pages={1,-3} undefined={2} overfull={3} warnings={4} missingchar={5}  [{6}]" -f `
            $Stem, $pages, $undefined, $overfull, $warnings, $missing, $status) -ForegroundColor $colour
}

Write-Host 'Refreshing paper/arxiv_submission/main.tex from paper/main.tex'
Copy-Item -LiteralPath (Join-Path $PaperDir 'main.tex') `
    -Destination (Join-Path $ArxivDir 'main.tex') -Force
$a = (Get-FileHash -LiteralPath (Join-Path $PaperDir 'main.tex') -Algorithm SHA256).Hash
$b = (Get-FileHash -LiteralPath (Join-Path $ArxivDir 'main.tex') -Algorithm SHA256).Hash
if ($a -ne $b) { throw 'SYNC FAILED: main.tex and arxiv_submission/main.tex differ' }
Write-Host '  SHA256 identical' -ForegroundColor Green

Write-Host 'English edition'
Invoke-Tectonic -WorkDir $PaperDir -Source 'main.tex'
Show-Build -WorkDir $PaperDir -Stem 'main'

Write-Host 'Chinese edition'
Invoke-Tectonic -WorkDir $PaperDir -Source 'main_zh.tex'
Show-Build -WorkDir $PaperDir -Stem 'main_zh'

if (-not $SkipArxiv) {
    Write-Host 'arXiv bundle'
    Invoke-Tectonic -WorkDir $ArxivDir -Source 'main.tex'
    Show-Build -WorkDir $ArxivDir -Stem 'main'
}

Write-Host 'Done.'
