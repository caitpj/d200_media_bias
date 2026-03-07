"""
analyse_secondary.py — SECONDARY ANALYSIS: All parties, news vs opinion.

Pipeline:
  Stage 1: Bias detection on all mentions (himel7/bias-detector)
  Stage 2: LLM classification on sampled biased mentions (Claude Sonnet 4.6)
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
    print(f"  4 parties, {MIN_YEAR}+, news + opinion")
    print("=" * 60)

    print("\nLoading mentions...")
    df = pd.read_csv(MENTIONS_PATH, sep="\t")
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["year"] = df["publish_date"].dt.year
    df["article_type"] = df["url"].apply(get_article_type)

    # Record total counts BEFORE bias detection (denominators for analysis)
    pre_counts = {}
    for party in ALL_PARTIES:
        for atype in ["news", "opinion"]:
            n = int(((df["party"] == party) &
                     (df["year"] >= MIN_YEAR) &
                     (df["article_type"] == atype)).sum())
            pre_counts[f"{party}_{atype}"] = n

    df = df[
        (df["party"].isin(ALL_PARTIES)) &
        (df["year"] >= MIN_YEAR) &
        (df["article_type"].isin(["news", "opinion"]))
    ].copy()

    pre_counts["total"] = int(len(df))
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
    # run_bias_pipeline returns (df_biased, counts)
    # counts has per-party {mentions, articles, biased} but not by article_type
    df_biased, _ = run_bias_pipeline(df, device)

    # Build detailed counts with article_type breakdown
    for party in ALL_PARTIES:
        for atype in ["news", "opinion"]:
            sub = df_biased[(df_biased["party"] == party) &
                            (df_biased["article_type"] == atype)]
            pre_counts[f"{party}_{atype}_biased"] = int(len(sub))

    # Save counts JSON
    counts_path = os.path.join(OUTPUT_DIR, "stage1_counts.json")
    with open(counts_path, "w") as f:
        json.dump(pre_counts, f, indent=2)
    print(f"  Saved: {counts_path}")

    # Save biased mentions
    biased_path = os.path.join(OUTPUT_DIR, "stage1_biased.tsv")
    df_biased.to_csv(biased_path, sep="\t", index=False)
    print(f"  Saved: {biased_path} ({len(df_biased)} biased mentions)")

    # Bias rates
    print("\n  Bias rates by party x article type:")
    print(f"  {'Party':<16s} {'news':>12s} {'opinion':>12s}")
    print(f"  {'-'*42}")
    for party in ALL_PARTIES:
        parts = []
        for atype in ["news", "opinion"]:
            n_total = pre_counts[f"{party}_{atype}"]
            n_b = pre_counts[f"{party}_{atype}_biased"]
            if n_total == 0:
                parts.append(f"{'n/a':>12s}")
            else:
                rate = n_b / n_total
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

    biased_path = os.path.join(OUTPUT_DIR, "stage1_biased.tsv")
    if not os.path.exists(biased_path):
        print(f"  ERROR: {biased_path} not found. Run --bias first.")
        return

    df_biased = pd.read_csv(biased_path, sep="\t")
    print(f"  {len(df_biased)} total biased mentions")

    # Sample up to LLM_MAX_PER_GROUP per party x article_type
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
        os.path.join(OUTPUT_DIR, "stage2_sampled.tsv"), sep="\t", index=False)

    results_df = run_llm_classification(df_sampled, OUTPUT_DIR)
    df_merged = merge_llm_results(df_sampled, results_df)

    df_on_target = df_merged[df_merged["on_target"] == 1].copy()
    n_off = (df_merged["on_target"] == 0).sum()
    print(f"\n  On-target: {len(df_on_target)}")
    print(f"  Off-target: {n_off} ({n_off/len(df_merged):.1%})")

    df_on_target.to_csv(
        os.path.join(OUTPUT_DIR, "stage2_on_target.tsv"), sep="\t",
        index=False)
    df_merged.to_csv(
        os.path.join(OUTPUT_DIR, "stage2_all.tsv"), sep="\t", index=False)


# ---------------------------------------------------------------------------
# Stage 3: Analysis
# ---------------------------------------------------------------------------

def stage_analyse():
    print("\n" + "=" * 60)
    print("SECONDARY STAGE 3: ANALYSIS")
    print("=" * 60)

    counts_path = os.path.join(OUTPUT_DIR, "stage1_counts.json")
    llm_path = os.path.join(OUTPUT_DIR, "stage2_llm_raw.tsv")
    sampled_path = os.path.join(OUTPUT_DIR, "stage2_sampled.tsv")

    if not os.path.exists(counts_path):
        print(f"  ERROR: {counts_path} not found. Run --bias first.")
        return
    if not os.path.exists(llm_path):
        print(f"  ERROR: {llm_path} not found. Run --llm first.")
        return

    with open(counts_path) as f:
        counts = json.load(f)

    df_sampled = pd.read_csv(sampled_path, sep="\t")
    df_llm = pd.read_csv(llm_path, sep="\t")
    df_biased = merge_llm_results(df_sampled, df_llm)
    df_on_target = df_biased[df_biased["on_target"] == 1].copy()

    # Save updated
    df_on_target.to_csv(
        os.path.join(OUTPUT_DIR, "stage2_on_target.tsv"), sep="\t",
        index=False)
    df_biased.to_csv(
        os.path.join(OUTPUT_DIR, "stage2_all.tsv"), sep="\t", index=False)

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
    # BIAS DETECTION RATES FROM COUNTS JSON
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
            n_total = counts.get(f"{party}_{atype}", 0)
            n_b = counts.get(f"{party}_{atype}_biased", 0)
            if n_total == 0:
                parts.extend(["     n/a", "    n/a"])
            else:
                rate = n_b / n_total
                parts.extend([f"{n_b:>8d}", f"{rate:>7.1%}"])
        print(f"  {party:<16s} {parts[0]} {parts[1]} {parts[2]} {parts[3]}")

    # ===================================================================
    # CHI-SQUARED ON BIAS RATES (from counts JSON)
    # ===================================================================

    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    for atype in ["news", "opinion"]:
        rows = []
        for party in ALL_PARTIES:
            n_total = counts.get(f"{party}_{atype}", 0)
            n_b = counts.get(f"{party}_{atype}_biased", 0)
            if n_total == 0:
                continue
            rows.append({
                "party": party,
                "biased": n_b,
                "non-biased": n_total - n_b,
            })
        if len(rows) < 2:
            continue

        contingency = pd.DataFrame(rows).set_index("party")
        chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
        n_total = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n_total * (min(contingency.shape) - 1)))

        print(f"\n  Chi-squared — {atype} bias rate across parties:")
        print(f"    chi2 = {chi2:.1f}, df = {dof}, p = {p_chi2:.4f}")
        print(f"    Cramer's V = {cramers_v:.3f}")

    # ===================================================================
    # SENTIMENT STRENGTH BREAKDOWN
    # ===================================================================

    print("\n" + "=" * 60)
    print("SENTIMENT STRENGTH BREAKDOWN (on-target)")
    print("=" * 60)

    # Discrete scale: {-1.0, -0.5, 0.0, +0.5, +1.0}
    bins = [
        ("Strong negative",   lambda s: s <= -0.75),
        ("Moderate negative",  lambda s: (s > -0.75) & (s < 0)),
        ("Neutral",            lambda s: s == 0.0),
        ("Moderate positive",  lambda s: (s > 0) & (s < 0.75)),
        ("Strong positive",    lambda s: s >= 0.75),
    ]

    print(f"\n  {'':20s}", end="")
    for party in ALL_PARTIES:
        print(f"  {party:>14s}", end="")
    print()
    print(f"  {'-'*78}")

    for label, cond in bins:
        print(f"  {label:20s}", end="")
        for party in ALL_PARTIES:
            subset = df_on_target[df_on_target["party"] == party]
            if len(subset) == 0:
                print(f"  {'n/a':>14s}", end="")
                continue
            count = cond(subset["sentiment_score"]).sum()
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