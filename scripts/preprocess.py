"""
preprocess.py — Extract party mentions with context windows from Nation.Cymru articles.

Pipeline: scrape.py → preprocess.py → analyse.py

Usage:
    python scripts/preprocess.py

Requirements:
    pip install pandas
"""

import json
import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                        "nation_cymru_articles.json")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Context window: sentences either side of a mention
CONTEXT_WINDOW = 1

# Skip captions and fragments shorter than this
MIN_SENTENCE_LENGTH = 30

# ---------------------------------------------------------------------------
# Party definitions
# ---------------------------------------------------------------------------

PARTIES = {
    "plaid_cymru": {
        "terms": ["Plaid", "Adam Price", "Leanne Wood", "Iorwerth"],
        "group": "left",
    },
    "reform_uk": {
        "terms": ["Reform", "Brexit Party", "Farage", "Dan Thomas"],
        "group": "right",
    },
    "labour": {
        "terms": ["Labour", "Eluned Morgan", "Gething", "Drakeford"],
        "group": "left",
    },
    "conservative": {
        "terms": ["Conservative", "RT Davies", "Tories", "Tory", "Darren Miller"],
        "group": "right",
    },
    "ukip": {
        "terms": ["UKIP"],
        "group": "right",
    },
}

# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting common abbreviations."""
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Rev)\.',
                  r'\1DOTPH', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.replace('DOTPH', '.') for s in sentences]
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def extract_mentions(sentences: list[str]) -> list[dict]:
    """
    Find all party mentions. For each, extract a context window.
    """
    mentions = []

    for party, config in PARTIES.items():
        for sent_idx, sentence in enumerate(sentences):
            if len(sentence) < MIN_SENTENCE_LENGTH:
                continue
            for term in config["terms"]:
                pos = sentence.find(term)
                if pos == -1:
                    continue

                # Context window
                start = max(0, sent_idx - CONTEXT_WINDOW)
                end = min(len(sentences), sent_idx + CONTEXT_WINDOW + 1)
                context = " ".join(sentences[start:end])

                mentions.append({
                    "party": party,
                    "group": config["group"],
                    "match": term,
                    "sentence": sentence,
                    "context": context,
                })
                break  # one match per party per sentence

    return mentions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load
    print(f"Loading {RAW_PATH}...")
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"  {len(articles)} articles loaded")

    # Process
    rows = []
    for article in articles:
        text = article.get("text", "")
        if not text or len(text) < 100:
            continue

        sentences = split_sentences(text)
        mentions = extract_mentions(sentences)

        for m in mentions:
            rows.append({
                "url": article["url"],
                "title": article["title"],
                "publish_date": article.get("publish_date"),
                "authors": ", ".join(article.get("authors", [])),
                **m,
            })

    df = pd.DataFrame(rows)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["year"] = df["publish_date"].dt.year

    # Summary
    print(f"\n  Total mentions: {len(df)}")
    print(f"\n  By party:")
    print(df["party"].value_counts().to_string(header=False))
    print(f"\n  By group:")
    print(df["group"].value_counts().to_string(header=False))
    print(f"\n  Date range: {df['publish_date'].min().date()} to "
          f"{df['publish_date'].max().date()}")

    # Save mentions
    mentions_path = os.path.join(PROCESSED_DIR, "party_mentions.tsv")
    df.to_csv(mentions_path, index=False, sep="\t")
    print(f"\n  Saved {mentions_path}")


if __name__ == "__main__":
    main()