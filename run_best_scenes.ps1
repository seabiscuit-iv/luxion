$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$listFile = Join-Path $root "best_scenes.txt"
$exe = Join-Path $root "build\bin\luxion.exe"

if (-not (Test-Path $listFile)) {
    Write-Host "Error: $listFile not found"
    exit 1
}
if (-not (Test-Path $exe)) {
    Write-Host "Error: $exe not found, build the project first"
    exit 1
}

$lines = Get-Content $listFile | Where-Object { $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") }

$failed = @()
$index = 0

Push-Location $root
try {

foreach ($line in $lines) {
    $index++
    $parts = $line.Trim() -split '\s+'

    for ($i = 0; $i -lt $parts.Length; $i++) {
        if (($parts[$i] -eq "-o" -or $parts[$i] -eq "--output") -and $i + 1 -lt $parts.Length) {
            $outDir = Split-Path -Parent (Join-Path $root $parts[$i + 1])
            if ($outDir -and -not (Test-Path $outDir)) {
                New-Item -ItemType Directory -Force -Path $outDir | Out-Null
            }
        }
    }   

    Write-Host ""
    Write-Host "[$index/$($lines.Count)] luxion $($parts -join ' ')"

    $start = Get-Date
    & $exe @parts
    $code = $LASTEXITCODE
    $elapsed = (Get-Date) - $start

    if ($code -ne 0) {
        Write-Host "  FAILED (exit $code) after $([int]$elapsed.TotalSeconds)s"
        $failed += $parts[0]
    }
    else {
        Write-Host "  done in $([int]$elapsed.TotalSeconds)s"
    }
}

}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Rendered $($lines.Count - $failed.Count)/$($lines.Count) scenes"
if ($failed.Count -gt 0) {
    Write-Host "Failed: $($failed -join ', ')"
    exit 1
}
