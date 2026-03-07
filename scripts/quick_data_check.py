"""
Quick script to get the TODO numbers for the report.
Run from the project root: python scripts/quick_data_check.py
"""
import json
import pandas as pd

# --- 1. Total articles ---
with open("data/raw/nation_cymru_articles.json") as f:
    articles = json.load(f)
print(f"Total articles: {len(articles)}")

# --- 2 & 3. Mention counts ---
df = pd.read_csv("data/processed/party_mentions.tsv", sep="\t")
df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
df["year"] = df["publish_date"].dt.year

# Article type from URL (same logic as ml_utils.get_article_type)
def get_article_type(url):
    if "/opinion/" in str(url):
        return "opinion"
    return "news"

df["article_type"] = df["url"].apply(get_article_type)

# Primary: Reform + Plaid, 2022+, news
primary = df[
    (df["party"].isin(["reform_uk", "plaid_cymru"])) &
    (df["year"] >= 2022) &
    (df["article_type"] == "news")
]
print(f"\nPrimary (Reform+Plaid, 2022-2026, news):")
print(f"  Total: {len(primary)}")
print(f"  Reform UK: {(primary['party'] == 'reform_uk').sum()}")
print(f"  Plaid Cymru: {(primary['party'] == 'plaid_cymru').sum()}")

# Secondary: 4 parties, 2022+, news + opinion
secondary = df[
    (df["party"].isin(["reform_uk", "plaid_cymru", "labour", "conservative"])) &
    (df["year"] >= 2022) &
    (df["article_type"].isin(["news", "opinion"]))
]
print(f"\nSecondary (4 parties, 2022-2026, news+opinion):")
print(f"  Total: {len(secondary)}")