"""
analyse.py — PRIMARY ANALYSIS: Reform UK vs Plaid Cymru head-to-head.

Pipeline:
  Stage 1: Bias detection (himel7/bias-detector)
  Stage 2: LLM entity-level sentiment attribution (Claude Haiku 4.5 API)
  Stage 3: Statistical analysis and reporting

Each stage is optional via flags. By default, all stages run.

Scope:
  - News articles only (excludes opinion pieces)
  - Editorial voice only (excludes quoted speech)
  - 2021 onwards (Reform UK's Welsh political presence)
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "primary")

FOCUS_PARTIES = ["reform_uk", "plaid_cymru"]
MIN_YEAR = 2022

SAMPLE_FRAC = 1.0
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Stage 1: Bias detection
# ---------------------------------------------------------------------------

def stage_bias():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STAGE 1: BIAS DETECTION")
    print(f"  Reform UK vs Plaid Cymru, news, editorial only, {MIN_YEAR}+")
    print("=" * 60)

    print("\nLoading mentions...")
    df = pd.read_csv(MENTIONS_PATH)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["year"] = df["publish_date"].dt.year
    df["article_type"] = df["url"].apply(get_article_type)

    print(f"  Total mentions: {len(df)}")

    df = df[
        (df["party"].isin(FOCUS_PARTIES)) &
        (df["year"] >= MIN_YEAR) &
        (df["article_type"] == "news") &
        (df["is_quote"] == False)
    ].copy()
    print(f"  After filtering: {len(df)}")

    for party in FOCUS_PARTIES:
        n = (df["party"] == party).sum()
        print(f"    {party}: {n}")

    # Sample
    df_sample = pd.concat([
        group.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
        for _, group in df.groupby("party")
    ]).reset_index(drop=True)

    print(f"  Sampled {len(df_sample)} mentions ({SAMPLE_FRAC:.0%})")
    for party in FOCUS_PARTIES:
        print(f"    {party}: {(df_sample['party'] == party).sum()}")

    device = get_device()
    df_all, df_biased = run_bias_pipeline(df_sample, device)

    all_path = os.path.join(OUTPUT_DIR, "primary_all.csv")
    df_all.to_csv(all_path, index=False)
    print(f"  Saved: {all_path}")

    print("\n  Bias rates:")
    for party in FOCUS_PARTIES:
        subset = df_all[df_all["party"] == party]
        n = len(subset)
        n_b = subset["is_biased"].sum()
        print(f"    {party}: {n_b}/{n} ({n_b/n:.1%})")


# ---------------------------------------------------------------------------
# Stage 2: LLM classification
# ---------------------------------------------------------------------------

def stage_llm():
    print("\n" + "=" * 60)
    print("STAGE 2: LLM CLASSIFICATION")
    print("=" * 60)

    all_path = os.path.join(OUTPUT_DIR, "primary_all.csv")
    if not os.path.exists(all_path):
        print(f"  ERROR: {all_path} not found. Run --bias first.")
        return

    df_all = pd.read_csv(all_path)
    df_biased = df_all[df_all["is_biased"] == 1].copy()
    print(f"  {len(df_biased)} biased mentions to classify")

    results_df = run_llm_classification(df_biased, OUTPUT_DIR)
    df_biased = merge_llm_results(df_biased, results_df)

    df_on_target = df_biased[df_biased["on_target"] == 1].copy()
    n_off = (df_biased["on_target"] == 0).sum()
    print(f"\n  On-target: {len(df_on_target)}")
    print(f"  Off-target: {n_off} ({n_off/len(df_biased):.1%})")

    biased_path = os.path.join(OUTPUT_DIR, "primary_biased.csv")
    df_on_target.to_csv(biased_path, index=False)
    print(f"  Saved: {biased_path}")

    full_path = os.path.join(OUTPUT_DIR, "biased_with_llm.csv")
    df_biased.to_csv(full_path, index=False)
    print(f"  Saved: {full_path}")


# ---------------------------------------------------------------------------
# Stage 3: Analysis
# ---------------------------------------------------------------------------

def stage_analyse():
    print("\n" + "=" * 60)
    print("STAGE 3: ANALYSIS")
    print("=" * 60)

    all_path = os.path.join(OUTPUT_DIR, "primary_all.csv")
    llm_path = os.path.join(OUTPUT_DIR, "llm_results.csv")

    if not os.path.exists(all_path):
        print(f"  ERROR: {all_path} not found. Run --bias first.")
        return
    if not os.path.exists(llm_path):
        print(f"  ERROR: {llm_path} not found. Run --llm first.")
        return

    df_all = pd.read_csv(all_path)
    df_all["publish_date"] = pd.to_datetime(df_all["publish_date"],
                                             errors="coerce")
    df_all["year"] = df_all["publish_date"].dt.year

    df_biased_raw = df_all[df_all["is_biased"] == 1].copy()
    df_biased = merge_llm_results(df_biased_raw, pd.read_csv(llm_path))

    df_on_target = df_biased[df_biased["on_target"] == 1].copy()
    n_off = (df_biased["on_target"] == 0).sum()

    # Save updated files
    biased_path = os.path.join(OUTPUT_DIR, "primary_biased.csv")
    df_on_target.to_csv(biased_path, index=False)

    full_path = os.path.join(OUTPUT_DIR, "biased_with_llm.csv")
    df_biased.to_csv(full_path, index=False)

    # ===================================================================
    # HEAD-TO-HEAD
    # ===================================================================

    print("\n" + "=" * 60)
    print("REFORM UK vs PLAID CYMRU — HEAD TO HEAD")
    print("=" * 60)

    for party in FOCUS_PARTIES:
        subset_all = df_all[df_all["party"] == party]
        subset_biased = df_on_target[df_on_target["party"] == party]
        n = len(subset_all)
        n_b = len(subset_biased)

        print(f"\n  {party}:")
        print(f"    Total mentions:       {n}")
        print(f"    Biased (on-target):   {n_b} ({n_b/n:.1%})")
        if n_b > 0:
            print(f"    Mean sentiment:       {subset_biased['sentiment_score'].mean():+.3f}")
            print(f"    Weighted bias score:  {subset_biased['weighted_bias_score'].mean():+.3f}")
            neg = (subset_biased["sentiment_score"] < 0).sum()
            pos = (subset_biased["sentiment_score"] > 0).sum()
            print(f"    Negative toward:      {neg} ({neg/n_b:.1%})")
            print(f"    Positive toward:      {pos} ({pos/n_b:.1%})")

    # ===================================================================
    # STATISTICAL TESTS
    # ===================================================================

    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    reform_all = df_all[df_all["party"] == "reform_uk"]
    plaid_all = df_all[df_all["party"] == "plaid_cymru"]
    reform_biased = df_on_target[df_on_target["party"] == "reform_uk"]
    plaid_biased = df_on_target[df_on_target["party"] == "plaid_cymru"]

    # 1. On-target bias rate: z-test
    n_r, n_p = len(reform_all), len(plaid_all)
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

        print(f"\n  2. Sentiment of on-target biased mentions (t-test):")
        print(f"     Reform: {s_r.mean():+.3f}  vs  Plaid: {s_p.mean():+.3f}")
        print(f"     t = {t_stat:+.3f}, p = {p_val_t:.4f}, Cohen's d = {d:+.3f}")
        print(f"     {'Significant at p<0.05' if p_val_t < 0.05 else 'Not significant'}")

    # 3. Weighted bias score: t-test
    if len(reform_biased) > 0 and len(plaid_biased) > 0:
        w_r = reform_biased["weighted_bias_score"]
        w_p = plaid_biased["weighted_bias_score"]
        t_stat_w, p_val_w = stats.ttest_ind(w_r, w_p)
        pooled_std_w = np.sqrt(
            ((len(w_r)-1)*w_r.std()**2 + (len(w_p)-1)*w_p.std()**2)
            / (len(w_r)+len(w_p)-2))
        d_w = (w_r.mean() - w_p.mean()) / pooled_std_w if pooled_std_w > 0 else 0

        print(f"\n  3. Weighted bias score (t-test):")
        print(f"     Reform: {w_r.mean():+.3f}  vs  Plaid: {w_p.mean():+.3f}")
        print(f"     t = {t_stat_w:+.3f}, p = {p_val_w:.4f}, Cohen's d = {d_w:+.3f}")
        print(f"     {'Significant at p<0.05' if p_val_w < 0.05 else 'Not significant'}")

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
    # COVERAGE VOLUME
    # ===================================================================

    print("\n" + "=" * 60)
    print("COVERAGE VOLUME")
    print("=" * 60)

    articles_reform = df_all[df_all["party"] == "reform_uk"]["url"].nunique()
    articles_plaid = df_all[df_all["party"] == "plaid_cymru"]["url"].nunique()
    print(f"\n  Unique articles mentioning Reform UK: {articles_reform}")
    print(f"  Unique articles mentioning Plaid Cymru: {articles_plaid}")
    print(f"  Ratio (Plaid/Reform): {articles_plaid/articles_reform:.1f}x")

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