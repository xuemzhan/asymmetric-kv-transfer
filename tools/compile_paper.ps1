$ErrorActionPreference = 'Stop'

$ToolsDir = $PSScriptRoot
$RepoDir = Split-Path -Parent $ToolsDir
$Tectonic = Join-Path $ToolsDir 'tectonic\tectonic.exe'
$PaperDir = Join-Path $RepoDir 'paper'

if (-not (Test-Path -LiteralPath $Tectonic)) {
    throw "Tectonic binary not found: $Tectonic"
}

Push-Location $PaperDir
try {
    & $Tectonic -X compile main.tex
    if ($LASTEXITCODE -ne 0) {
        throw "tectonic failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
