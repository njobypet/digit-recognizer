# Downloads the MNIST dataset into the data/ directory

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DataDir = Join-Path $ScriptDir "..\data"

if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
}

$BaseUrl = "https://ossci-datasets.s3.amazonaws.com/mnist"

$Files = @(
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
)

foreach ($f in $Files) {
    $extracted = $f -replace '\.gz$', ''
    $extractedPath = Join-Path $DataDir $extracted
    $gzPath = Join-Path $DataDir $f

    if (-not (Test-Path $extractedPath)) {
        Write-Host "Downloading $f..."
        Invoke-WebRequest -Uri "$BaseUrl/$f" -OutFile $gzPath

        Write-Host "Extracting $f..."
        $inStream = [System.IO.File]::OpenRead($gzPath)
        $gzStream = New-Object System.IO.Compression.GzipStream($inStream, [System.IO.Compression.CompressionMode]::Decompress)
        $outStream = [System.IO.File]::Create($extractedPath)
        $gzStream.CopyTo($outStream)
        $outStream.Close()
        $gzStream.Close()
        $inStream.Close()

        Remove-Item $gzPath
    } else {
        Write-Host "$extracted already exists, skipping."
    }
}

Write-Host ""
Write-Host "MNIST dataset downloaded to: $DataDir"
Get-ChildItem $DataDir
