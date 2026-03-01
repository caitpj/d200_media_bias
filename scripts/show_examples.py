"""
show_examples.py — Display example biased sentences with NLI verification.

Reads from data/primary/primary_biased.csv (Reform UK vs Plaid Cymru,
2021+, news only, verified on-target). Also reads the raw biased mentions
before NLI filtering to show examples of what was filtered out.

Usage:
    python scripts/show_examples.py
"""

import os
import pandas as pd

PRIMARY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "primary")
BIASED_PATH = os.path.join(PRIMARY_DIR, "primary_biased.csv")
ALL_PATH = os.path.join(PRIMARY_DIR, "primary_all.csv")

PARTIES = ["reform_uk", "plaid_cymru"]
N_EXAMPLES = 5


def show_on_target(df, party, direction, n=N_EXAMPLES):
    """Show verified on-target examples."""
    subset = df[
        (df["party"] == party) &
        (df["bias_direction"] == direction)
    ].copy()

    if direction == "negative":
        subset = subset.sort_values("sentiment_score", ascending=True)
    else:
        subset = subset.sort_values("sentiment_score", ascending=False)

    print(f"\n  {'─' * 70}")
    print(f"  {party.upper()} — ON-TARGET {direction.upper()} bias "
          f"({len(subset)} total, showing {min(n, len(subset))})")
    print(f"  {'─' * 70}")

    for i, (_, row) in enumerate(subset.head(n).iterrows()):
        crit = row.get('nli_critical_score', 0)
        supp = row.get('nli_supportive_score', 0)
        print(f"\n  [{i+1}] Sent: {row['sentiment_score']:+.3f} "
              f"| Weighted: {row['weighted_bias_score']:+.3f} "
              f"| NLI crit: {crit:.3f} supp: {supp:.3f} "
              f"| Quote: {row['is_quote']}")
        print(f"  Title: {str(row['title'])[:80]}")
        print(f"  Sentence: {str(row['sentence'])[:200]}")
        if len(str(row['sentence'])) > 200:
            print(f"            {str(row['sentence'])[200:400]}")
        print(f"  URL: {row['url']}")


def show_filtered_out(df_all, df_biased, party, n=N_EXAMPLES):
    """Show examples that were filtered out by NLI (off-target)."""
    # Biased but not in the verified set
    biased_all = df_all[
        (df_all["party"] == party) &
        (df_all["is_biased"] == 1)
    ].copy()

    # Find those NOT in df_biased (filtered out by NLI)
    if "nli_on_target" in df_all.columns:
        # If NLI columns are in df_all already
        off_target = biased_all[biased_all.get("nli_on_target", True) == False]
    else:
        # Match by url+sentence to find what's in biased_all but not df_biased
        biased_keys = set(zip(
            df_biased[df_biased["party"] == party]["url"],
            df_biased[df_biased["party"] == party]["sentence"]
        ))
        off_target = biased_all[
            ~biased_all.apply(
                lambda r: (r["url"], r["sentence"]) in biased_keys, axis=1
            )
        ]

    print(f"\n  {'─' * 70}")
    print(f"  {party.upper()} — FILTERED OUT (off-target) "
          f"({len(off_target)} total, showing {min(n, len(off_target))})")
    print(f"  These were biased but NOT directed at {party}")
    print(f"  {'─' * 70}")

    for i, (_, row) in enumerate(off_target.head(n).iterrows()):
        print(f"\n  [{i+1}] Quote: {row['is_quote']}")
        print(f"  Title: {str(row['title'])[:80]}")
        print(f"  Sentence: {str(row['sentence'])[:200]}")
        if len(str(row['sentence'])) > 200:
            print(f"            {str(row['sentence'])[200:400]}")
        print(f"  URL: {row['url']}")


def main():
    print("Loading primary analysis results...")
    df_biased = pd.read_csv(BIASED_PATH)
    df_all = pd.read_csv(ALL_PATH)

    median_sent = df_biased["sentiment_score"].median()
    df_biased["bias_direction"] = "neutral"
    df_biased.loc[df_biased["sentiment_score"] < median_sent, "bias_direction"] = "negative"
    df_biased.loc[df_biased["sentiment_score"] > median_sent, "bias_direction"] = "positive"

    print(f"  {len(df_biased)} verified on-target biased mentions")
    print(f"  Median sentiment: {median_sent:+.3f}")
    for party in PARTIES:
        n = (df_biased["party"] == party).sum()
        n_all_biased = ((df_all["party"] == party) & (df_all["is_biased"] == 1)).sum()
        print(f"  {party}: {n} on-target (of {n_all_biased} biased)")

    # On-target examples
    for party in PARTIES:
        for direction in ["negative", "positive"]:
            show_on_target(df_biased, party, direction)

    # Filtered-out examples
    print("\n\n" + "=" * 60)
    print("EXAMPLES FILTERED OUT BY NLI (off-target)")
    print("=" * 60)

    for party in PARTIES:
        show_filtered_out(df_all, df_biased, party)

    print()


if __name__ == "__main__":
    main()