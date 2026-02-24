"""
scrape.py — Collect Welsh political news articles from Nation.Cymru.

Uses the WordPress REST API (wp-json) for reliable, structured data going
back to Nation.Cymru's 2017 launch.

Usage:
    python scripts/scrape.py

Requirements:
    pip install requests beautifulsoup4
"""

import json
import os
import time
from datetime import datetime
from html import unescape

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
BASE_URL = "https://nation.cymru/wp-json/wp/v2/posts"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; research project)"}

# Pull everything from Nation.Cymru's launch
CUTOFF_DATE = datetime(2017, 1, 1)

# Targeted search terms covering all major parties + general politics
SEARCH_TERMS = [
    # General
    "Senedd",
    "election",
    "Welsh government",
    # Plaid Cymru (nationalist)
    "Plaid Cymru",
    "Plaid",
    "Rhun ap Iorwerth",
    "Adam Price",
    "Leanne Wood",
    # Reform UK / Brexit Party
    "Reform UK",
    "Reform",
    "Brexit Party",
    "Farage",
    "Reform Wales",
    # Labour
    "Welsh Labour",
    "Labour Wales",
    "Eluned Morgan",
    "Mark Drakeford",
    "Vaughan Gething",
    # Conservatives
    "Welsh Conservative",
    "Conservative Wales",
    "Andrew RT Davies",
]

# Welsh politics keywords — at least 2 must match
POLITICS_KEYWORDS = [
    # Institutions & general
    "senedd", "assembly", "election", "welsh government", "first minister",
    "devolution", "wales politics", "ms ", " ms,",
    # Plaid Cymru
    "plaid cymru", "plaid", "rhun ap iorwerth", "adam price", "leanne wood",
    # Reform UK / Brexit Party
    "reform uk", "reform party", "brexit party", "farage", "nigel farage",
    # Labour
    "welsh labour", "labour wales", "eluned morgan", "mark drakeford",
    "vaughan gething",
    # Conservatives
    "welsh conservative", "conservative wales", "andrew rt davies",
    "tory", "tories",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_html(html_text: str) -> str:
    """Remove HTML tags from text."""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def is_welsh_politics(text: str, title: str = "") -> bool:
    """Check if text is about Welsh politics (at least 2 keyword matches)."""
    combined = (text + " " + title).lower()
    matches = sum(1 for kw in POLITICS_KEYWORDS if kw in combined)
    return matches >= 2


# Author cache: WP author ID → name
_author_cache = {}


def get_author(author_id: int) -> str:
    """Resolve a WordPress author ID to a name."""
    if author_id in _author_cache:
        return _author_cache[author_id]
    try:
        resp = requests.get(
            f"https://nation.cymru/wp-json/wp/v2/users/{author_id}",
            headers=HEADERS, timeout=30,
        )
        if resp.status_code == 200:
            name = resp.json().get("name", "")
            _author_cache[author_id] = name
            return name
    except Exception:
        pass
    _author_cache[author_id] = ""
    return ""


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------

def scrape() -> list:
    articles = []
    seen_ids = set()

    for search_term in SEARCH_TERMS:
        page = 1
        consecutive_failures = 0
        label = search_term or "all posts"

        while True:
            params = {
                "per_page": 20,
                "page": page,
                "orderby": "date",
                "order": "desc",
                "after": CUTOFF_DATE.isoformat() + "Z",
            }
            if search_term:
                params["search"] = search_term

            print(f"  [{label}] page {page}...")

            # Retry up to 3 times
            posts = None
            for attempt in range(3):
                try:
                    resp = requests.get(BASE_URL, params=params,
                                        headers=HEADERS, timeout=60)
                    if resp.status_code == 400:
                        posts = []
                        break
                    resp.raise_for_status()
                    posts = resp.json()
                    consecutive_failures = 0
                    break
                except Exception as e:
                    wait = (attempt + 1) * 10
                    print(f"    Attempt {attempt + 1} failed: {e}")
                    print(f"    Retrying in {wait}s...")
                    time.sleep(wait)

            if posts is None:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    print(f"  Too many failures, moving on")
                    break
                page += 1
                continue

            if not posts:
                break

            for post in posts:
                post_id = post["id"]
                if post_id in seen_ids:
                    continue
                seen_ids.add(post_id)

                raw_html = post.get("content", {}).get("rendered", "")
                text = strip_html(raw_html)
                title = unescape(
                    strip_html(post.get("title", {}).get("rendered", ""))
                )

                if len(text) < 100:
                    continue
                if not is_welsh_politics(text, title):
                    continue

                author_id = post.get("author", 0)
                author_name = get_author(author_id) if author_id else ""

                articles.append({
                    "url": post.get("link", ""),
                    "title": title,
                    "text": text,
                    "authors": [author_name] if author_name else [],
                    "publish_date": post.get("date", None),
                    "source": "nation_cymru",
                    "wp_id": post_id,
                })

            print(f"    {len(articles)} relevant ({len(seen_ids)} seen)")

            page += 1
            time.sleep(3)

            if page > 100:
                break

    return articles


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Scraping Nation.Cymru from {CUTOFF_DATE.strftime('%Y-%m-%d')}")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}\n")

    articles = scrape()

    output_path = os.path.join(OUTPUT_DIR, "nation_cymru_articles.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nDone. {len(articles)} articles saved to {output_path}")


if __name__ == "__main__":
    main()