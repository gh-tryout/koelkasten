# Bouwt dist\Koelkastanalyse\Koelkastanalyse.exe (geen Python nodig op de doel-pc).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m PyInstaller --noconfirm --clean koelkastanalyse.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller mislukte." }

$dist = Join-Path $PSScriptRoot "dist\Koelkastanalyse"
$doc = Join-Path $PSScriptRoot "documentatie_koelkastanalyse.html"
if (Test-Path $doc) {
    Copy-Item $doc (Join-Path $dist "documentatie_koelkastanalyse.html") -Force
}

@"
Koelkastanalyse EN 62552-3
==========================

Start: dubbelklik Koelkastanalyse.exe
Er opent een zwart venster en daarna de browser (http://127.0.0.1:...).
Sluit het zwarte venster om te stoppen.

Deze map moet bij elkaar blijven (vooral de map _internal).
Meetdata zit niet in deze map: kies in de GUI de power-map en per
meetpunt de temperatuurmap op de pc.

Windows Defender kan de exe de eerste keer blokkeren omdat die niet
digitaal ondertekend is. Kies dan 'Toch uitvoeren'.

Python of Cursor is niet nodig.
"@ | Set-Content -Path (Join-Path $dist "LEESMIJ.txt") -Encoding UTF8

Write-Host "Klaar: $dist\Koelkastanalyse.exe"
