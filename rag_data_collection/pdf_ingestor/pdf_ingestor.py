"""
PDF Ingestor — ingests local PDFs + downloads from arXiv automatically

Install:
  pip install pymupdf arxiv boto3 tqdm requests

Run:
  # Ingest local folder of PDFs
  python pdf_ingestor.py --mode local --input ./pdfs --output ./raw_docs

  # Download + ingest from arXiv by search query
  python pdf_ingestor.py --mode arxiv --query "prompt engineering LLM" --max 30 --output ./raw_docs

  # Both
  python pdf_ingestor.py --mode both --input ./pdfs --query "chain of thought prompting" --output ./raw_docs
"""

import os
import re
import json
import time
import hashlib
import argparse
import requests
import tempfile
from pathlib import Path
from datetime import datetime
import fitz                 # pymupdf
import arxiv
import boto3
from tqdm import tqdm
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

# ── arXiv search queries for PromptBridge knowledge base ─────────────────────
ARXIV_QUERIES = [
    "prompt engineering large language models",
    "chain of thought reasoning LLM",
    "retrieval augmented generation RAG",
    "instruction tuning language models",
    "few shot learning prompting",
    "LLM alignment RLHF reward model",
    "hallucination mitigation language models",
]

# ── PDF text extraction ───────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: str) -> dict:
    """
    Extract clean text from a PDF file using PyMuPDF.
    Returns structured doc with per-section metadata.
    """
    doc = fitz.open(pdf_path)
    full_text  = []
    sections   = []
    char_count = 0

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")   # returns (x0,y0,x1,y1,text,block_no,block_type)
        page_text = ""
        for block in blocks:
            if block[6] == 0:              # type 0 = text (not image)
                text = block[4].strip()
                if text:
                    page_text += text + "\n"

        # Detect section headers (short lines in ALL CAPS or title-like)
        for line in page_text.split("\n"):
            stripped = line.strip()
            if (
                len(stripped) > 3 and len(stripped) < 80
                and (stripped.isupper() or re.match(r"^\d+\.?\s+[A-Z]", stripped))
            ):
                sections.append({"page": page_num + 1, "heading": stripped})

        full_text.append(page_text)
        char_count += len(page_text)

    raw_text = "\n".join(full_text)

    # Clean: remove excessive whitespace, page numbers, headers/footers
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
    raw_text = re.sub(r"(\b\d{1,3}\b)\n", "", raw_text)    # lone page numbers
    raw_text = re.sub(r"[ \t]{3,}", " ", raw_text)

    # Extract title from first page (heuristic: longest short line near top)
    first_page_lines = [l.strip() for l in full_text[0].split("\n") if l.strip()]
    title_candidates = [l for l in first_page_lines[:8] if 10 < len(l) < 120]
    title = max(title_candidates, key=len) if title_candidates else Path(pdf_path).stem

    doc.close()

    return {
        "raw_text":   raw_text,
        "title":      title,
        "pages":      len(doc),
        "sections":   sections,
        "char_count": char_count,
    }

def pdf_to_doc(pdf_path: str, source: str = "local", metadata: dict = None) -> dict | None:
    """Convert a PDF file into a structured document dict."""
    path = Path(pdf_path)
    try:
        extracted = extract_pdf_text(str(path))
        if extracted["char_count"] < 500:
            print(f"  Skipping {path.name}: too short ({extracted['char_count']} chars)")
            return None

        doc_id = hashlib.md5(path.name.encode()).hexdigest()[:12]
        return {
            "id":           doc_id,
            "source":       source,
            "title":        extracted["title"],
            "text":         extracted["raw_text"],
            "pages":        extracted["pages"],
            "sections":     extracted["sections"],
            "char_count":   extracted["char_count"],
            "filename":     path.name,
            "ingested_at":  datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

    except Exception as e:
        print(f"  Error reading {path.name}: {e}")
        return None

# ── Local PDF ingestion ───────────────────────────────────────────────────────
def ingest_local(input_dir: str, output_dir: Path) -> list[dict]:
    pdf_files = list(Path(input_dir).rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {input_dir}")

    out = output_dir / "pdf_local"
    out.mkdir(parents=True, exist_ok=True)
    docs = []

    for pdf_path in tqdm(pdf_files, desc="Ingesting local PDFs"):
        doc = pdf_to_doc(str(pdf_path), source="local_pdf")
        if doc:
            (out / f"{doc['id']}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2)
            )
            docs.append(doc)

    return docs

# ── arXiv download + ingest ───────────────────────────────────────────────────
def ingest_arxiv(query: str, max_results: int, output_dir: Path) -> list[dict]:
    out = output_dir / "pdf_arxiv"
    out.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=50, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    docs = []
    print(f"Fetching up to {max_results} papers for: '{query}'")

    for paper in tqdm(client.results(search), desc="Downloading arXiv PDFs", total=max_results):
        try:
            # Check if already downloaded
            safe_id = paper.entry_id.split("/")[-1].replace("/", "_")
            out_file = out / f"{safe_id}.json"
            if out_file.exists():
                docs.append(json.loads(out_file.read_text()))
                continue

            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                paper.download_pdf(filename=tmp.name)
                doc = pdf_to_doc(
                    tmp.name,
                    source="arxiv",
                    metadata={
                        "arxiv_id":   safe_id,
                        "url":        paper.entry_id,
                        "authors":    [a.name for a in paper.authors[:5]],
                        "published":  paper.published.isoformat() if paper.published else None,
                        "abstract":   paper.summary[:500],
                        "categories": paper.categories,
                        "title":      paper.title,   # override extracted title with official
                    }
                )
                os.unlink(tmp.name)

            if doc:
                out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
                docs.append(doc)

            time.sleep(1.0)   # arXiv rate limit

        except Exception as e:
            print(f"  Failed {paper.entry_id}: {e}")
            continue

    return docs

def ingest_arxiv_multi(queries: list[str], max_per_query: int, output_dir: Path) -> list[dict]:
    """Run multiple arXiv queries and deduplicate by arxiv_id."""
    all_docs = []
    seen_ids = set()
    for query in queries:
        docs = ingest_arxiv(query, max_per_query, output_dir)
        for doc in docs:
            arxiv_id = doc.get("arxiv_id", doc["id"])
            if arxiv_id not in seen_ids:
                seen_ids.add(arxiv_id)
                all_docs.append(doc)
    return all_docs

# ── S3 upload ─────────────────────────────────────────────────────────────────
def upload_to_s3(output_dir: Path, bucket: str):
    s3    = boto3.client("s3")
    files = list(output_dir.rglob("*.json"))
    print(f"Uploading {len(files)} files to s3://{bucket}/raw_docs/pdf/")
    for f in tqdm(files):
        key = f"raw_docs/pdf/{f.relative_to(output_dir)}"
        s3.upload_file(str(f), bucket, str(key))

# ── Summary ───────────────────────────────────────────────────────────────────
def write_summary(output_dir: Path, all_docs: list[dict]):
    arxiv_docs = [d for d in all_docs if d.get("source") == "arxiv"]
    local_docs = [d for d in all_docs if d.get("source") == "local_pdf"]

    summary = {
        "total_docs":    len(all_docs),
        "arxiv_docs":    len(arxiv_docs),
        "local_docs":    len(local_docs),
        "total_chars":   sum(d["char_count"] for d in all_docs),
        "total_pages":   sum(d.get("pages", 0) for d in all_docs),
        "ingested_at":   datetime.utcnow().isoformat(),
    }
    (output_dir / "pdf_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n── PDF Ingestion Summary ─────────────────────────────")
    print(f"  Total docs   : {summary['total_docs']}")
    print(f"  arXiv papers : {summary['arxiv_docs']}")
    print(f"  Local PDFs   : {summary['local_docs']}")
    print(f"  Total chars  : {summary['total_chars']:,}")
    print(f"  Total pages  : {summary['total_pages']}")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["local", "arxiv", "both"], default="both")
    parser.add_argument("--input",     default="./pdfs",       help="Local PDF folder")
    parser.add_argument("--query",     default="",             help="Single arXiv query (overrides defaults)")
    parser.add_argument("--max",       type=int, default=20,   help="Max papers per arXiv query")
    parser.add_argument("--output",    default="./raw_docs")
    parser.add_argument("--s3_bucket", default="")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_docs = []

    if args.mode in ("local", "both"):
        all_docs += ingest_local(args.input, output_dir)

    if args.mode in ("arxiv", "both"):
        queries = [args.query] if args.query else ARXIV_QUERIES
        all_docs += ingest_arxiv_multi(queries, args.max, output_dir)

    write_summary(output_dir, all_docs)

    if args.s3_bucket:
        upload_to_s3(output_dir, args.s3_bucket)

if __name__ == "__main__":
    main()
