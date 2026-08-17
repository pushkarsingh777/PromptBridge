#!/bin/bash
# PromptBridge RAG — Full Data Collection Pipeline
# Run this script to collect everything in one shot.
#
# Usage:
#   export ANTHROPIC_API_KEY=your_key
#   export AWS_ACCESS_KEY_ID=your_key      # optional, for S3
#   export AWS_SECRET_ACCESS_KEY=your_key  # optional, for S3
#   chmod +x run_pipeline.sh && ./run_pipeline.sh

set -e   # stop on any error

echo ""
echo "══════════════════════════════════════════════════"
echo "  PromptBridge RAG — Data Collection Pipeline"
echo "══════════════════════════════════════════════════"
echo ""

S3_BUCKET="${S3_BUCKET:-}"   # set env var to enable S3 upload
OUTPUT="./data"

# ── Step 1: Web scraping ──────────────────────────────────────────────────────
echo "▶ Step 1/3: Web scraping..."
python scrapers/web_scraper.py \
  --sites all \
  --output "$OUTPUT/raw_docs" \
  --max_pages 100 \
  ${S3_BUCKET:+--s3_bucket "$S3_BUCKET"}

echo ""
echo "✓ Web scraping complete"
echo ""

# ── Step 2: arXiv PDF ingestion ───────────────────────────────────────────────
echo "▶ Step 2/3: arXiv PDF ingestion..."
python pdf_ingestor/pdf_ingestor.py \
  --mode arxiv \
  --max 15 \
  --output "$OUTPUT/raw_docs" \
  ${S3_BUCKET:+--s3_bucket "$S3_BUCKET"}

# If you have local PDFs, uncomment:
# python pdf_ingestor/pdf_ingestor.py \
#   --mode local \
#   --input ./my_pdfs \
#   --output "$OUTPUT/raw_docs"

echo ""
echo "✓ PDF ingestion complete"
echo ""

# ── Step 3: Synthetic pair generation ─────────────────────────────────────────
echo "▶ Step 3/3: Generating synthetic training pairs..."
python synthetic_pairs/generate_pairs.py \
  --input "$OUTPUT/raw_docs" \
  --output "$OUTPUT/training_pairs" \
  --chunk_tokens 400 \
  --overlap_tokens 60 \
  --pairs_per_chunk 3 \
  --max_chunks_per_doc 15 \
  --hard_negatives \
  ${S3_BUCKET:+--s3_bucket "$S3_BUCKET"}

echo ""
echo "══════════════════════════════════════════════════"
echo "  ✓ Data collection complete!"
echo ""
echo "  Output structure:"
echo "  data/"
echo "  ├── raw_docs/            ← scraped web pages + PDFs"
echo "  │   ├── promptingguide/  "
echo "  │   ├── learnprompting/  "
echo "  │   ├── pdf_arxiv/       "
echo "  │   └── summary.json     "
echo "  └── training_pairs/      ← ready for embedding fine-tune"
echo "      ├── corpus.jsonl     ← all chunks → vector DB"
echo "      ├── train_pairs.jsonl"
echo "      └── val_pairs.jsonl  "
echo "══════════════════════════════════════════════════"
