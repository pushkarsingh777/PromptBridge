param (
    [switch]$UploadR2 = ($env:UPLOAD_R2 -eq "true" -or $args -contains "--r2")
)

$ErrorActionPreference = "Stop"

# Ensure script executes in its directory
Set-Location -Path $PSScriptRoot

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  PromptBridge RAG - Data Collection Pipeline"     -ForegroundColor Cyan
Write-Host "  Cloud Storage: Cloudflare R2"                    -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

if ($UploadR2) {
    Write-Host ">> Cloudflare R2 Upload: ENABLED`n" -ForegroundColor Green
} else {
    Write-Host ">> Cloudflare R2 Upload: DISABLED (saving locally only)`n" -ForegroundColor DarkGray
}

$Output = "./data"

# -- Step 1: Web scraping ------------------------------------------------------
Write-Host ">> Step 1/3: Web scraping..." -ForegroundColor Yellow
$scraperArgs = @(
    "scrapers/web_scraper.py",
    "--sites", "all",
    "--output", "$Output/raw_docs",
    "--max_pages", "100"
)
if ($UploadR2) { $scraperArgs += "--upload_r2" }   # removed --s3_bucket

python @scraperArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Web scraping failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "`n[OK] Web scraping complete`n" -ForegroundColor Green

# -- Step 2: arXiv PDF ingestion -----------------------------------------------
Write-Host ">> Step 2/3: arXiv PDF ingestion..." -ForegroundColor Yellow
$pdfArgs = @(
    "pdf_ingestor/pdf_ingestor.py",
    "--mode", "arxiv",
    "--max", "15",
    "--output", "$Output/raw_docs"
)
if ($UploadR2) { $pdfArgs += "--upload_r2" }       # removed --s3_bucket

python @pdfArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] PDF ingestion failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "`n[OK] PDF ingestion complete`n" -ForegroundColor Green

# -- Step 3: Synthetic pair generation -----------------------------------------
Write-Host ">> Step 3/3: Generating synthetic training pairs..." -ForegroundColor Yellow
$pairArgs = @(
    "synthetic_pairs/generate_pairs.py",
    "--input",           "$Output/raw_docs",
    "--output",          "$Output/training_pairs",
    "--chunk_tokens",    "400",
    "--overlap_tokens",  "60",
    "--pairs_per_chunk", "3",
    "--max_chunks_per_doc", "15",
    "--hard_negatives"
)
if ($UploadR2) { $pairArgs += "--upload_r2" }       # removed --s3_bucket

python @pairArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Synthetic pair generation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  [SUCCESS] Data collection complete!"             -ForegroundColor Green
Write-Host ""
Write-Host "  Output structure:"
Write-Host "  data/"
Write-Host "  +-- raw_docs/            <- scraped web pages + PDFs"
Write-Host "  |   +-- promptingguide/"
Write-Host "  |   +-- learnprompting/"
Write-Host "  |   +-- pdf_arxiv/"
Write-Host "  |   \-- summary.json"
Write-Host "  \-- training_pairs/      <- ready for embedding fine-tune"
Write-Host "      +-- corpus.jsonl     <- all chunks -> vector DB"
Write-Host "      +-- train_pairs.jsonl"
Write-Host "      \-- val_pairs.jsonl"
Write-Host "==================================================" -ForegroundColor Cyan