import pandas as pd

df = pd.read_csv("data/processed/party_mentions.csv")
df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
df["year"] = df["publish_date"].dt.year

def get_article_type(url):
    if "/opinion/" in str(url):
        return "opinion"
    return "news"

df["article_type"] = df["url"].apply(get_article_type)

secondary = df[
    (df["party"].isin(["reform_uk", "plaid_cymru", "labour", "conservative"])) &
    (df["year"] >= 2022) &
    (df["article_type"].isin(["news", "opinion"])) &
    (df["is_quote"] == False)
]
print(f"Secondary mentions (editorial voice): {len(secondary)}")