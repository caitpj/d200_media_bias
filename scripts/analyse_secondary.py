"""
analyse_secondary.py — SECONDARY ANALYSIS: All parties, news vs opinion.

Pipeline:
  Stage 1: Bias detection on all mentions (himel7/bias-detector)
  Stage 2: LLM classification on sampled biased mentions (Claude Haiku 4.5)
  Stage 3: Statistical analysis comparing parties and article types

To limit API costs, Stage 2 samples up to LLM_MAX_PER_GROUP biased
mentions per party per article type.

Scope:
  - 4 parties: Reform UK, Plaid Cymru, Labour, Conservative
  - 2022 onwards
  - News and opinion articles analysed separately

Usage:
    python scripts/analyse_secondary.py                # All stages
    python scripts/analyse_secondary.py --bias         # Stage 1 only
    python scripts/analyse_secondary.py --llm          # Stage 2 only
    python scripts/analyse_secondary.py --analyse      # Stage 3 only
"""

import os
import sys
import argparse

import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from ml_utils import (
    get_device, get_article_type, run_bias_pipeline,
    run_llm_classification, merge_llm_results,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MENTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                             "processed", "party_mentions.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "secondary")

ALL_PARTIES = ["reform_uk", "plaid_cymru", "labour", "conservative"]
MIN_YEAR = 2022

# Max biased mentions to send to LLM per party per article type
LLM_MAX_PER_GROUP = 250

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Stage 1: Bias detection
# ---------------------------------------------------------------------------

def stage_bias():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("SECONDARY STAGE 1: BIAS DETECTION")
    print(f"  4 parties, {MIN_YEAR}+, news + opinion, editorial voice")
    print("=" * 60)

    print("\nLoading mentions...")
    df = pd.read_csv(MENTIONS_PATH)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["year"] = df["publish_date"].dt.year
    df["article_type"] = df["url"].apply(get_article_type)

    df = df[
        (df["party"].isin(ALL_PARTIES)) &
        (df["year"] >= MIN_YEAR) &
        (df["article_type"].isin(["news", "opinion"])) &
        (df["is_quote"] == False)
    ].copy()

    print(f"  {len(df)} mentions after filtering")
    print(f"\n  By party:")
    for party in ALL_PARTIES:
        n = (df["party"] == party).sum()
        print(f"    {party}: {n}")
    print(f"\n  By article type:")
    for atype in ["news", "opinion"]:
        n = (df["article_type"] == atype).sum()
        print(f"    {atype}: {n}")

    device = get_device()
    df_all, df_biased = run_bias_pipeline(df, device)

    df_all.to_csv(os.path.join(OUTPUT_DIR, "secondary_all.csv"), index=False)
    print(f"  Saved: secondary_all.csv")

    # Bias rates
    print("\n  Bias rates by party × article type:")
    print(f"  {'Party':<16s} {'news':>12s} {'opinion':>12s}")
    print(f"  {'-'*42}")
    for party in ALL_PARTIES:
        parts = []
        for atype in ["news", "opinion"]:
            sub = df_all[(df_all["party"] == party) &
                         (df_all["article_type"] == atype)]
            if len(sub) == 0:
                parts.append(f"{'n/a':>12s}")
            else:
                rate = sub["is_biased"].mean()
                n_b = sub["is_biased"].sum()
                parts.append(f"{n_b:>4d} ({rate:>5.1%})")
        print(f"  {party:<16s} {''.join(parts)}")


# ---------------------------------------------------------------------------
# Stage 2: LLM classification (with budget cap)
# ---------------------------------------------------------------------------

def stage_llm():
    print("\n" + "=" * 60)
    print("SECONDARY STAGE 2: LLM CLASSIFICATION")
    print(f"  Max {LLM_MAX_PER_GROUP} per party per article type")
    print("=" * 60)

    all_path = os.path.join(OUTPUT_DIR, "secondary_all.csv")
    if not os.path.exists(all_path):
        print(f"  ERROR: {all_path} not found. Run --bias first.")
        return

    df_all = pd.read_csv(all_path)
    df_biased = df_all[df_all["is_biased"] == 1].copy()
    print(f"  {len(df_biased)} total biased mentions")

    # Sample up to LLM_MAX_PER_GROUP per party × article_type
    sampled = []
    print(f"\n  Sampling for LLM:")
    for party in ALL_PARTIES:
        for atype in ["news", "opinion"]:
            group = df_biased[
                (df_biased["party"] == party) &
                (df_biased["article_type"] == atype)
            ]
            n = min(len(group), LLM_MAX_PER_GROUP)
            if n == 0:
                continue
            sample = group.sample(n=n, random_state=RANDOM_SEED)
            sampled.append(sample)
            print(f"    {party} / {atype}: {n} of {len(group)}")

    df_sampled = pd.concat(sampled).reset_index(drop=True)
    df_sampled["id"] = range(len(df_sampled))
    print(f"  Total for LLM: {len(df_sampled)}")

    # Save the sampled biased mentions (pre-LLM)
    df_sampled.to_csv(
        os.path.join(OUTPUT_DIR, "biased_sampled.csv"), index=False)

    results_df = run_llm_classification(df_sampled, OUTPUT_DIR)
    df_merged = merge_llm_results(df_sampled, results_df)

    df_on_target = df_merged[df_merged["on_target"] == 1].copy()
    n_off = (df_merged["on_target"] == 0).sum()
    print(f"\n  On-target: {len(df_on_target)}")
    print(f"  Off-target: {n_off} ({n_off/len(df_merged):.1%})")

    df_on_target.to_csv(
        os.path.join(OUTPUT_DIR, "secondary_biased.csv"), index=False)
    df_merged.to_csv(
        os.path.join(OUTPUT_DIR, "biased_with_llm.csv"), index=False)


# ---------------------------------------------------------------------------
# Stage 3: Analysis
# ---------------------------------------------------------------------------

def stage_analyse():
    print("\n" + "=" * 60)
    print("SECONDARY STAGE 3: ANALYSIS")
    print("=" * 60)

    all_path = os.path.join(OUTPUT_DIR, "secondary_all.csv")
    llm_path = os.path.join(OUTPUT_DIR, "llm_results.csv")
    sampled_path = os.path.join(OUTPUT_DIR, "biased_sampled.csv")

    if not os.path.exists(all_path):
        print(f"  ERROR: {all_path} not found. Run --bias first.")
        return
    if not os.path.exists(llm_path):
        print(f"  ERROR: {llm_path} not found. Run --llm first.")
        return

    df_all = pd.read_csv(all_path)
    df_sampled = pd.read_csv(sampled_path)
    df_biased = merge_llm_results(df_sampled, pd.read_csv(llm_path))
    df_on_target = df_biased[df_biased["on_target"] == 1].copy()

    # Save updated
    df_on_target.to_csv(
        os.path.join(OUTPUT_DIR, "secondary_biased.csv"), index=False)
    df_biased.to_csv(
        os.path.join(OUTPUT_DIR, "biased_with_llm.csv"), index=False)

    # ===================================================================
    # ALL PARTIES SUMMARY
    # ===================================================================

    print("\n" + "=" * 60)
    print("ALL PARTIES — ON-TARGET BIAS (sampled biased mentions)")
    print("=" * 60)

    print(f"\n  {'Party':<16s} {'biased':>8s} {'on-tgt':>8s} "
          f"{'on%':>7s} {'sent':>7s} {'neg%':>7s} {'pos%':>7s}")
    print(f"  {'-'*62}")

    for party in ALL_PARTIES:
        p_biased = df_biased[df_biased["party"] == party]
        p_on = df_on_target[df_on_target["party"] == party]
        n_b = len(p_biased)
        n_on = len(p_on)
        on_pct = n_on / n_b if n_b > 0 else 0
        sent = p_on["sentiment_score"].mean() if n_on > 0 else 0
        neg = (p_on["sentiment_score"] < 0).sum() / n_on if n_on > 0 else 0
        pos = (p_on["sentiment_score"] > 0).sum() / n_on if n_on > 0 else 0
        print(f"  {party:<16s} {n_b:>8d} {n_on:>8d} "
              f"{on_pct:>6.1%} {sent:>+7.3f} {neg:>6.1%} {pos:>6.1%}")

    # ===================================================================
    # NEWS vs OPINION COMPARISON
    # ===================================================================

    print("\n" + "=" * 60)
    print("NEWS vs OPINION — ON-TARGET BIAS")
    print("=" * 60)

    for atype in ["news", "opinion"]:
        at_biased = df_biased[df_biased["article_type"] == atype]
        at_on = df_on_target[df_on_target["article_type"] == atype]

        print(f"\n  --- {atype.upper()} ---")
        print(f"  {'Party':<16s} {'biased':>8s} {'on-tgt':>8s} "
              f"{'on%':>7s} {'sent':>7s} {'neg%':>7s} {'pos%':>7s}")
        print(f"  {'-'*62}")

        for party in ALL_PARTIES:
            p_b = at_biased[at_biased["party"] == party]
            p_on = at_on[at_on["party"] == party]
            n_b = len(p_b)
            n_on = len(p_on)
            if n_b == 0:
                continue
            on_pct = n_on / n_b
            sent = p_on["sentiment_score"].mean() if n_on > 0 else 0
            neg = (p_on["sentiment_score"] < 0).sum() / n_on if n_on > 0 else 0
            pos = (p_on["sentiment_score"] > 0).sum() / n_on if n_on > 0 else 0
            print(f"  {party:<16s} {n_b:>8d} {n_on:>8d} "
                  f"{on_pct:>6.1%} {sent:>+7.3f} {neg:>6.1%} {pos:>6.1%}")

    # ===================================================================
    # BIAS RATES FROM FULL DATA (not sampled)
    # ===================================================================

    print("\n" + "=" * 60)
    print("BIAS DETECTION RATES (full data, Stage 1)")
    print("=" * 60)

    print(f"\n  {'Party':<16s} {'news_n':>8s} {'news%':>8s} "
          f"{'opin_n':>8s} {'opin%':>8s}")
    print(f"  {'-'*50}")

    for party in ALL_PARTIES:
        parts = []
        for atype in ["news", "opinion"]:
            sub = df_all[(df_all["party"] == party) &
                         (df_all["article_type"] == atype)]
            if len(sub) == 0:
                parts.extend(["n/a", "n/a"])
            else:
                n_b = sub["is_biased"].sum()
                rate = sub["is_biased"].mean()
                parts.extend([f"{n_b:>8d}", f"{rate:>7.1%}"])
        print(f"  {party:<16s} {parts[0]} {parts[1]} {parts[2]} {parts[3]}")

    # ===================================================================
    # CHI-SQUARED ON BIAS RATES
    # ===================================================================

    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    for atype in ["news", "opinion"]:
        at = df_all[df_all["article_type"] == atype]
        if len(at) == 0:
            continue
        contingency = pd.crosstab(at["party"], at["bias_label"])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue
        chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
        n_total = len(at)
        cramers_v = np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1)))

        print(f"\n  Chi-squared — {atype} bias rate across parties:")
        print(f"    χ² = {chi2:.1f}, df = {dof}, p = {p_chi2:.4f}")
        print(f"    Cramér's V = {cramers_v:.3f}")

    # ===================================================================
    # SENTIMENT STRENGTH BREAKDOWN
    # ===================================================================

    print("\n" + "=" * 60)
    print("SENTIMENT STRENGTH BREAKDOWN (on-target)")
    print("=" * 60)

    bins = [
        ("Strong negative", -1.01, -0.75),
        ("Moderate negative", -0.75, -0.0),
        ("Neutral/weak", -0.0, 0.0),
        ("Moderate positive", 0.0, 0.75),
        ("Strong positive", 0.75, 1.01),
    ]

    print(f"\n  {'':20s}", end="")
    for party in ALL_PARTIES:
        print(f"  {party:>14s}", end="")
    print()
    print(f"  {'-'*78}")

    for label, lo, hi in bins:
        print(f"  {label:20s}", end="")
        for party in ALL_PARTIES:
            subset = df_on_target[df_on_target["party"] == party]
            if len(subset) == 0:
                print(f"  {'n/a':>14s}", end="")
                continue
            if label == "Neutral/weak":
                count = (subset["sentiment_score"] == 0).sum()
            else:
                count = ((subset["sentiment_score"] > lo) &
                         (subset["sentiment_score"] <= hi)).sum()
            pct = count / len(subset)
            print(f"  {count:>4d} ({pct:>5.1%})", end="")
        print()

    print("\n  Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Secondary analysis: all parties, news vs opinion")
    parser.add_argument("--bias", action="store_true",
                        help="Run Stage 1: bias detection")
    parser.add_argument("--llm", action="store_true",
                        help="Run Stage 2: LLM classification")
    parser.add_argument("--analyse", action="store_true",
                        help="Run Stage 3: statistical analysis")
    args = parser.parse_args()

    run_all = not (args.bias or args.llm or args.analyse)

    if run_all or args.bias:
        stage_bias()
    if run_all or args.llm:
        stage_llm()
    if run_all or args.analyse:
        stage_analyse()


if __name__ == "__main__":
    main()