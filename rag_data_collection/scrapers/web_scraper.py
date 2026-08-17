"""
Web Scraper — collects prompt engineering documentation
Targets: promptingguide.ai, learnprompting.org, Anthropic docs, OpenAI docs

Install:
  pip install requests beautifulsoup4 markdownify trafilatura boto3 tqdm

Run:
  python web_scraper.py --sites all --output ./raw_docs
  python web_scraper.py --sites promptingguide --output ./raw_docs
"""

import os
import re
import json
import time
import hashlib
import argparse
import requests
import boto3
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from collections import deque
from tqdm import tqdm
from bs4 import BeautifulSoup
import trafilatura          # extracts clean main text from any webpage
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

# ── Target sites ──────────────────────────────────────────────────────────────
SITES = {
    "promptingguide": {
        "seed": "https://www.promptingguide.ai",
        "allowed_domain": "promptingguide.ai",
        "max_pages": 300,
    },
    "learnprompting": {
        "seed": "https://learnprompting.org/docs/intro",
        "allowed_domain": "learnprompting.org",
        "max_pages": 300,
    },
    "anthropic_docs": {
        "seed": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview",
        "allowed_domain": "docs.anthropic.com",
        "max_pages": 150,
    },
    "openai_cookbook": {
        "seed": "https://cookbook.openai.com",
        "allowed_domain": "cookbook.openai.com",
        "max_pages": 150,
    },
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PromptBridge-RAG-Bot/1.0; research use)",
    "Accept": "text/html,application/xhtml+xml",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def url_to_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]

def clean_text(raw_html: str) -> str | None:
    """Use trafilatura to extract main content, fall back to BS4."""
    text = trafilatura.extract(
        raw_html,
        include_tables=True,
        include_links=False,
        include_comments=False,
    )
    if text and len(text.strip()) > 200:
        return text.strip()

    # Fallback: BS4
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) > 200 else None

def extract_links(html: str, base_url: str, allowed_domain: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        if (
            parsed.scheme in ("http", "https")
            and allowed_domain in parsed.netloc
            and "#" not in href
            and not href.endswith((".png", ".jpg", ".pdf", ".zip", ".mp4"))
        ):
            links.append(href.split("#")[0])
    return list(set(links))

# ── Core crawler ──────────────────────────────────────────────────────────────
def crawl_site(name: str, config: dict, output_dir: Path) -> list[dict]:
    site_dir = output_dir / name
    site_dir.mkdir(parents=True, exist_ok=True)

    queue     = deque([config["seed"]])
    visited   = set()
    collected = []
    session   = requests.Session()
    session.headers.update(HEADERS)

    pbar = tqdm(desc=f"Crawling {name}", total=config["max_pages"], unit="pages")

    while queue and len(collected) < config["max_pages"]:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue

            html = resp.text
            text = clean_text(html)
            if not text:
                continue

            # Extract title
            soup  = BeautifulSoup(html, "html.parser")
            title = (soup.find("h1") or soup.find("title") or soup.find("h2"))
            title = title.get_text(strip=True) if title else url

            doc = {
                "id":     url_to_id(url),
                "url":    url,
                "source": name,
                "title":  title,
                "text":   text,
                "scraped_at": datetime.utcnow().isoformat(),
                "char_count": len(text),
            }

            # Save individual file
            out_file = site_dir / f"{doc['id']}.json"
            out_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
            collected.append(doc)

            # Enqueue new links
            new_links = extract_links(html, url, config["allowed_domain"])
            for link in new_links:
                if link not in visited:
                    queue.append(link)

            pbar.update(1)
            time.sleep(0.5)   # polite crawl delay

        except Exception as e:
            print(f"\nSkipped {url}: {e}")
            continue

    pbar.close()
    print(f"  {name}: collected {len(collected)} pages")
    return collected

# ── S3 upload ─────────────────────────────────────────────────────────────────
def upload_to_s3(output_dir: Path, bucket: str):
    s3 = boto3.client("s3")
    files = list(output_dir.rglob("*.json"))
    print(f"Uploading {len(files)} files to s3://{bucket}/raw_docs/web/...")
    for f in tqdm(files, desc="Uploading"):
        key = f"raw_docs/web/{f.relative_to(output_dir)}"
        s3.upload_file(str(f), bucket, str(key))
    print("Upload complete.")

# ── Summary ───────────────────────────────────────────────────────────────────
def write_summary(output_dir: Path, all_docs: list[dict]):
    summary = {
        "total_docs":   len(all_docs),
        "total_chars":  sum(d["char_count"] for d in all_docs),
        "by_source":    {},
        "scraped_at":   datetime.utcnow().isoformat(),
    }
    for doc in all_docs:
        s = doc["source"]
        summary["by_source"].setdefault(s, {"count": 0, "chars": 0})
        summary["by_source"][s]["count"] += 1
        summary["by_source"][s]["chars"] += doc["char_count"]

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n── Scraping Summary ─────────────────────────────────")
    print(f"  Total docs  : {summary['total_docs']}")
    print(f"  Total chars : {summary['total_chars']:,}")
    for src, info in summary["by_source"].items():
        print(f"  {src:<20}: {info['count']} docs / {info['chars']:,} chars")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites",      default="all",
                        help="all | promptingguide | learnprompting | anthropic_docs | openai_cookbook")
    parser.add_argument("--output",     default="./raw_docs")
    parser.add_argument("--s3_bucket",  default="",    help="Upload to S3 if set")
    parser.add_argument("--max_pages",  type=int, default=0,
                        help="Override max_pages per site (0 = use site default)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sites_to_run = (
        SITES if args.sites == "all"
        else {k: v for k, v in SITES.items() if k in args.sites.split(",")}
    )

    if not sites_to_run:
        print(f"Unknown site(s): {args.sites}. Choose from: {list(SITES.keys())}")
        return

    all_docs = []
    for name, config in sites_to_run.items():
        if args.max_pages:
            config["max_pages"] = args.max_pages
        docs = crawl_site(name, config, output_dir)
        all_docs.extend(docs)

    write_summary(output_dir, all_docs)

    if args.s3_bucket:
        upload_to_s3(output_dir, args.s3_bucket)

if __name__ == "__main__":
    main()
