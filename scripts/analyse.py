"""
analyse.py — PRIMARY ANALYSIS: Reform UK vs Plaid Cymru head-to-head.

Framework (D'Alessio & Allen, 2000):
  - Coverage bias: differential mention volumes
  - Statement bias: favourability of editorial framing (two-stage pipeline)

Pipeline:
  Stage 1: Bias detection (himel7/bias-detector) — captures linguistic bias
  Stage 2: LLM entity-level sentiment attribution (Claude Sonnet 4.6) — resolves statement bias
  Stage 3: Statistical analysis and reporting

Each stage is optional via flags. By default, all stages run.

Scope:
  - News articles only (excludes opinion pieces)
  - 2022 onwards (Reform UK's Welsh political presence)
  - Focus on Reform UK and Plaid Cymru

Usage:
    python scripts/analyse.py                  # Run all stages
    python scripts/analyse.py --bias           # Stage 1 only
    python scripts/analyse.py --llm            # Stage 2 only (needs Stage 1 output)
    python scripts/analyse.py --analyse        # Stage 3 only (needs Stage 1+2 output)
    python scripts/analyse.py --bias --llm     # Stages 1+2
    python scripts/analyse.py --llm --analyse  # Stages 2+3
"""

import os
import sys
import json
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
                             "processed", "party_mentions.tsv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "primary")

FOCUS_PARTIES = ["reform_uk", "plaid_cymru"]
MIN_YEAR = 2022


# ---------------------------------------------------------------------------
# Stage 1: Bias detection
# ---------------------------------------------------------------------------

def stage_bias():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STAGE 1: BIAS DETECTION")
    print(f"  Reform UK vs Plaid Cymru, news, {MIN_YEAR}+")
    print("=" * 60)

    print("\nLoading mentions...")
    df = pd.read_csv(MENTIONS_PATH, sep="\t")
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["year"] = df["publish_date"].dt.year
    df["article_type"] = df["url"].apply(get_article_type)

    print(f"  Total mentions: {len(df)}")

    df = df[
        (df["party"].isin(FOCUS_PARTIES)) &
        (df["year"] >= MIN_YEAR) &
        (df["article_type"] == "news")
    ].copy()
    print(f"  After filtering: {len(df)}")

    for party in FOCUS_PARTIES:
        n = (df["party"] == party).sum()
        print(f"    {party}: {n}")

    device = get_device()
    df_biased, counts = run_bias_pipeline(df, device)

    biased_path = os.path.join(OUTPUT_DIR, "stage1_biased.tsv")
    df_biased.to_csv(biased_path, index=False, sep="\t")
    print(f"  Saved: {biased_path}")

    counts_path = os.path.join(OUTPUT_DIR, "stage1_counts.json")
    with open(counts_path, "w") as f:
        json.dump(counts, f, indent=2)
    print(f"  Saved: {counts_path}")

    print("\n  Bias rates:")
    for party in FOCUS_PARTIES:
        c = counts[party]
        print(f"    {party}: {c['biased']}/{c['mentions']} "
              f"({c['biased']/c['mentions']:.1%})")


# ---------------------------------------------------------------------------
# Stage 2: LLM classification
# ---------------------------------------------------------------------------

def stage_llm():
    print("\n" + "=" * 60)
    print("STAGE 2: LLM CLASSIFICATION")
    print("=" * 60)

    biased_path = os.path.join(OUTPUT_DIR, "stage1_biased.tsv")
    if not os.path.exists(biased_path):
        print(f"  ERROR: {biased_path} not found. Run --bias first.")
        return

    df_biased = pd.read_csv(biased_path, sep="\t")
    print(f"  {len(df_biased)} biased mentions to classify")

    results_df = run_llm_classification(df_biased, OUTPUT_DIR)
    df_merged = merge_llm_results(df_biased, results_df)

    n_on = (df_merged["on_target"] == 1).sum()
    n_off = (df_merged["on_target"] == 0).sum()
    print(f"\n  On-target: {n_on}")
    print(f"  Off-target: {n_off} ({n_off/len(df_merged):.1%})")


# ---------------------------------------------------------------------------
# Stage 3: Analysis
# ---------------------------------------------------------------------------

def stage_analyse():
    print("\n" + "=" * 60)
    print("STAGE 3: ANALYSIS")
    print("=" * 60)

    biased_path = os.path.join(OUTPUT_DIR, "stage1_biased.tsv")
    llm_path = os.path.join(OUTPUT_DIR, "stage2_llm_raw.tsv")
    counts_path = os.path.join(OUTPUT_DIR, "stage1_counts.json")

    for path, label in [(biased_path, "--bias"), (llm_path, "--llm"),
                        (counts_path, "--bias")]:
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found. Run {label} first.")
            return

    df_biased_raw = pd.read_csv(biased_path, sep="\t")
    df_biased = merge_llm_results(df_biased_raw, pd.read_csv(llm_path, sep="\t"))
    df_on_target = df_biased[df_biased["on_target"] == 1].copy()

    with open(counts_path, "r") as f:
        counts = json.load(f)

    # ===================================================================
    # COVERAGE BIAS (D'Alessio & Allen, 2000)
    # ===================================================================

    print("\n" + "=" * 60)
    print("COVERAGE BIAS")
    print("=" * 60)

    for party in FOCUS_PARTIES:
        c = counts[party]
        mpa = c["mentions"] / c["articles"] if c["articles"] > 0 else 0
        print(f"\n  {party}:")
        print(f"    Mentions:              {c['mentions']}")
        print(f"    Unique articles:       {c['articles']}")
        print(f"    Mentions per article:  {mpa:.1f}")

    reform_mentions = counts["reform_uk"]["mentions"]
    plaid_mentions = counts["plaid_cymru"]["mentions"]
    reform_articles = counts["reform_uk"]["articles"]
    plaid_articles = counts["plaid_cymru"]["articles"]

    print(f"\n  Mention ratio (Plaid/Reform): {plaid_mentions/reform_mentions:.2f}x")
    print(f"  Article ratio (Plaid/Reform): {plaid_articles/reform_articles:.2f}x")

    # Chi-squared: are mentions evenly split between parties?
    total_mentions = reform_mentions + plaid_mentions
    chi2_cov = (reform_mentions - total_mentions/2)**2 / (total_mentions/2) + \
               (plaid_mentions - total_mentions/2)**2 / (total_mentions/2)
    p_cov = 1 - stats.chi2.cdf(chi2_cov, df=1)
    print(f"\n  Chi-squared (equal coverage): χ² = {chi2_cov:.1f}, "
          f"df = 1, p = {p_cov:.4f}")

    # ===================================================================
    # STATEMENT BIAS (D'Alessio & Allen, 2000)
    # ===================================================================

    print("\n" + "=" * 60)
    print("STATEMENT BIAS")
    print("=" * 60)

    for party in FOCUS_PARTIES:
        n = counts[party]["mentions"]
        subset_biased = df_on_target[df_on_target["party"] == party]
        n_b = len(subset_biased)

        print(f"\n  {party}:")
        print(f"    Total mentions:       {n}")
        print(f"    Biased (on-target):   {n_b} ({n_b/n:.1%})")
        if n_b > 0:
            print(f"    Mean sentiment:       {subset_biased['sentiment_score'].mean():+.3f}")
            neg = (subset_biased["sentiment_score"] < 0).sum()
            pos = (subset_biased["sentiment_score"] > 0).sum()
            print(f"    Negative toward:      {neg} ({neg/n_b:.1%})")
            print(f"    Positive toward:      {pos} ({pos/n_b:.1%})")

    # ===================================================================
    # LLM TARGET ATTRIBUTION
    # ===================================================================

    print("\n" + "=" * 60)
    print("LLM TARGET ATTRIBUTION")
    print("=" * 60)

    for party in FOCUS_PARTIES:
        p_biased = df_biased[df_biased["party"] == party]
        p_on = df_on_target[df_on_target["party"] == party]
        n_b = len(p_biased)
        n_on = len(p_on)
        on_pct = n_on / n_b if n_b > 0 else 0
        print(f"  {party}: {n_on}/{n_b} on-target ({on_pct:.1%})")

    total_biased = len(df_biased)
    total_off = (df_biased["on_target"] == 0).sum()
    print(f"\n  Total off-target filtered: {total_off}/{total_biased} "
          f"({total_off/total_biased:.1%})")

    # ===================================================================
    # STATISTICAL TESTS
    # ===================================================================

    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    reform_biased = df_on_target[df_on_target["party"] == "reform_uk"]
    plaid_biased = df_on_target[df_on_target["party"] == "plaid_cymru"]

    # 1. On-target bias rate: z-test
    n_r = counts["reform_uk"]["mentions"]
    n_p = counts["plaid_cymru"]["mentions"]
    b_r = len(reform_biased)
    b_p = len(plaid_biased)
    r_r, r_p = b_r / n_r, b_p / n_p
    p_pooled = (b_r + b_p) / (n_r + n_p)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_r + 1/n_p))
    z = (r_r - r_p) / se if se > 0 else 0
    p_val_z = 2 * (1 - stats.norm.cdf(abs(z)))
    h = 2 * (np.arcsin(np.sqrt(r_r)) - np.arcsin(np.sqrt(r_p)))

    print(f"\n  1. On-target bias rate (z-test for proportions):")
    print(f"     Reform: {r_r:.1%}  vs  Plaid: {r_p:.1%}")
    print(f"     z = {z:+.3f}, p = {p_val_z:.4f}, Cohen's h = {h:+.3f}")
    print(f"     {'Significant at p<0.05' if p_val_z < 0.05 else 'Not significant'}")

    # 2. Sentiment: t-test
    if len(reform_biased) > 0 and len(plaid_biased) > 0:
        s_r = reform_biased["sentiment_score"]
        s_p = plaid_biased["sentiment_score"]
        t_stat, p_val_t = stats.ttest_ind(s_r, s_p)
        pooled_std = np.sqrt(
            ((len(s_r)-1)*s_r.std()**2 + (len(s_p)-1)*s_p.std()**2)
            / (len(s_r)+len(s_p)-2))
        d = (s_r.mean() - s_p.mean()) / pooled_std if pooled_std > 0 else 0

        print(f"\n  2. Statement bias — sentiment (t-test):")
        print(f"     Reform: {s_r.mean():+.3f}  vs  Plaid: {s_p.mean():+.3f}")
        print(f"     t = {t_stat:+.3f}, p = {p_val_t:.4f}, Cohen's d = {d:+.3f}")
        print(f"     {'Significant at p<0.05' if p_val_t < 0.05 else 'Not significant'}")

    # ===================================================================
    # SENTIMENT STRENGTH BREAKDOWN
    # ===================================================================

    print("\n" + "=" * 60)
    print("SENTIMENT STRENGTH BREAKDOWN (on-target biased mentions)")
    print("=" * 60)

    bins = [
        ("Strong negative", -1.01, -0.75),
        ("Moderate negative", -0.75, -0.0),
        ("Neutral/weak", -0.0, 0.0),
        ("Moderate positive", 0.0, 0.75),
        ("Strong positive", 0.75, 1.01),
    ]

    print(f"\n  {'':20s}", end="")
    for party in FOCUS_PARTIES:
        print(f"  {party:>14s}", end="")
    print()
    print(f"  {'-'*50}")

    for label, lo, hi in bins:
        print(f"  {label:20s}", end="")
        for party in FOCUS_PARTIES:
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

    print(f"\n  Strong signal ratio (strong neg / strong pos):")
    for party in FOCUS_PARTIES:
        subset = df_on_target[df_on_target["party"] == party]
        if len(subset) == 0:
            continue
        strong_neg = (subset["sentiment_score"] <= -0.75).sum()
        strong_pos = (subset["sentiment_score"] >= 0.75).sum()
        ratio = strong_neg / strong_pos if strong_pos > 0 else float('inf')
        print(f"    {party}: {strong_neg} neg vs {strong_pos} pos "
              f"(ratio: {ratio:.1f}x)")

    # ===================================================================
    # EXAMPLE REASONING
    # ===================================================================

    print("\n" + "=" * 60)
    print("EXAMPLE LLM CLASSIFICATIONS")
    print("=" * 60)

    for party in FOCUS_PARTIES:
        on = df_on_target[df_on_target["party"] == party]
        off = df_biased[(df_biased["party"] == party) &
                        (df_biased["on_target"] == 0)]

        print(f"\n  {party} — on-target examples:")
        for i, (_, row) in enumerate(on.head(3).iterrows()):
            print(f"    [{i+1}] sent={row['sentiment_score']:+.1f} "
                  f"| {row['reasoning']}")
            print(f"        {str(row['sentence'])[:120]}")

        if len(off) > 0:
            print(f"\n  {party} — filtered out (off-target):")
            for i, (_, row) in enumerate(off.head(3).iterrows()):
                print(f"    [{i+1}] {row['reasoning']}")
                print(f"        {str(row['sentence'])[:120]}")

    print("\n  Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Primary analysis: Reform UK vs Plaid Cymru")
    parser.add_argument("--bias", action="store_true",
                        help="Run Stage 1: bias detection")
    parser.add_argument("--llm", action="store_true",
                        help="Run Stage 2: LLM classification")
    parser.add_argument("--analyse", action="store_true",
                        help="Run Stage 3: statistical analysis")
    args = parser.parse_args()

    # Default: run all stages if no flags
    run_all = not (args.bias or args.llm or args.analyse)

    if run_all or args.bias:
        stage_bias()
    if run_all or args.llm:
        stage_llm()
    if run_all or args.analyse:
        stage_analyse()


if __name__ == "__main__":
    main()