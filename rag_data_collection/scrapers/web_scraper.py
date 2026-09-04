"""
Web Scraper — collects prompt engineering documentation
Targets: promptingguide.ai, learnprompting.org, Anthropic docs, OpenAI docs

Install:
  pip install requests beautifulsoup4 trafilatura boto3 tqdm python-dotenv

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
import trafilatura

# ── Load .env ──────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
# Finds .env in current folder or any parent folder automatically
load_dotenv()

# ── Validate R2 credentials ───────────────────────────────────────────────────
def check_r2_credentials():
    missing = []
    for key in ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]:
        if not os.getenv(key):
            missing.append(key)
    if missing:
        print(f"\n[R2 ERROR] Missing in .env file: {', '.join(missing)}")
        print("Open your .env file and fill in these values.")
        print("Then try uploading again.\n")
        return False
    return True

def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )

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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def url_to_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path   = parsed.path.rstrip("/")
    if "docs.anthropic.com" in parsed.netloc and path.startswith("/docs/en/"):
        path = "/en/docs/" + path[len("/docs/en/"):]
    return f"{parsed.scheme}://{parsed.netloc}{path}"

def clean_text(raw_html: str) -> str | None:
    text = trafilatura.extract(
        raw_html,
        include_tables=True,
        include_links=False,
        include_comments=False,
    )
    if text and len(text.strip()) > 200:
        return text.strip()
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) > 200 else None

def extract_links(html: str, base_url: str, allowed_domain: str) -> list[str]:
    soup  = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href   = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        if (
            parsed.scheme in ("http", "https")
            and allowed_domain in parsed.netloc
            and not href.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf", ".zip", ".mp4"))
        ):
            norm_url = normalize_url(href.split("#")[0].split("?")[0])
            links.append(norm_url)
    return list(set(links))

# ── Core crawler ──────────────────────────────────────────────────────────────
def crawl_site(name: str, config: dict, output_dir: Path) -> list[dict]:
    site_dir = output_dir / name
    site_dir.mkdir(parents=True, exist_ok=True)

    seed_url  = normalize_url(config["seed"])
    queue     = deque([seed_url])
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
            resp = session.get(url, timeout=15, allow_redirects=True)
            if resp.url != url:
                visited.add(normalize_url(resp.url))
            if resp.status_code != 200:
                continue
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue

            html  = resp.text
            text  = clean_text(html)
            if not text:
                continue

            soup  = BeautifulSoup(html, "html.parser")
            title = soup.find("h1") or soup.find("title") or soup.find("h2")
            title = title.get_text(strip=True) if title else url

            doc = {
                "id":         url_to_id(url),
                "url":        url,
                "source":     name,
                "title":      title,
                "text":       text,
                "scraped_at": datetime.utcnow().isoformat(),
                "char_count": len(text),
            }

            (site_dir / f"{doc['id']}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            collected.append(doc)

            for link in extract_links(html, url, config["allowed_domain"]):
                if link not in visited:
                    queue.append(link)

            pbar.update(1)
            time.sleep(0.5)

        except requests.exceptions.TooManyRedirects:
            continue
        except Exception as e:
            print(f"\nSkipped {url}: {e}")
            continue

    pbar.close()
    print(f"  {name}: collected {len(collected)} pages")
    return collected

# ── R2 Upload ─────────────────────────────────────────────────────────────────
def upload_to_r2(output_dir: Path):
    if not check_r2_credentials():
        return

    bucket = os.getenv("R2_BUCKET_NAME")
    r2     = get_r2_client()
    files  = list(output_dir.rglob("*.json"))

    print(f"\nUploading {len(files)} files to R2 bucket: {bucket}")

    failed = []
    for f in tqdm(files, desc="Uploading to R2"):
        # ── FIX: use forward slashes for R2 key (Windows produces backslashes) ──
        relative = f.relative_to(output_dir)
        key      = "raw_docs/web/" + "/".join(relative.parts)   # always forward slashes

        try:
            r2.upload_file(str(f), bucket, key)
        except Exception as e:
            failed.append((str(f), str(e)))

    if failed:
        print(f"\n[WARNING] {len(failed)} files failed to upload:")
        for path, err in failed[:5]:
            print(f"  {path}: {err}")
    else:
        print(f"[R2] Upload complete. {len(files)} files in r2://{bucket}/raw_docs/web/")

# ── Summary ───────────────────────────────────────────────────────────────────
def write_summary(output_dir: Path, all_docs: list[dict]):
    summary = {
        "total_docs":  len(all_docs),
        "total_chars": sum(d["char_count"] for d in all_docs),
        "by_source":   {},
        "scraped_at":  datetime.utcnow().isoformat(),
    }
    for doc in all_docs:
        s = doc["source"]
        summary["by_source"].setdefault(s, {"count": 0, "chars": 0})
        summary["by_source"][s]["count"] += 1
        summary["by_source"][s]["chars"] += doc["char_count"]

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n── Scraping Summary ─────────────────────────────────")
    print(f"  Total docs  : {summary['total_docs']}")
    print(f"  Total chars : {summary['total_chars']:,}")
    for src, info in summary["by_source"].items():
        print(f"  {src:<20}: {info['count']} docs / {info['chars']:,} chars")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites",     default="all",
                        help="all | promptingguide | learnprompting | anthropic_docs | openai_cookbook")
    parser.add_argument("--output",    default="./raw_docs")
    parser.add_argument("--upload_r2", action="store_true", help="Upload to Cloudflare R2")
    parser.add_argument("--max_pages", type=int, default=0,
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
        all_docs.extend(crawl_site(name, config, output_dir))

    write_summary(output_dir, all_docs)

    if args.upload_r2:
        upload_to_r2(output_dir)

if __name__ == "__main__":
    main()