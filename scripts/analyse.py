"""
analyse_v2.py — Combined bias and sentiment analysis on party mentions.

Runs two models on the same sample:
  1. himel7/bias-detector — sentence-level media bias detection (biased/non-biased)
  2. cardiffnlp/twitter-roberta-base-sentiment-latest — general sentiment (neg/neu/pos)

Combines both to determine presence, rate, and direction of media bias.

Usage:
    python scripts/analyse_v2.py

Requirements:
    pip install pandas torch transformers sentencepiece protobuf
"""

import os
import time

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MENTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                             "processed", "party_mentions.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

BIAS_MODEL = "himel7/bias-detector"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

TARGET_PARTIES = ["reform_uk", "plaid_cymru", "labour", "conservative", "ukip"]

SAMPLE_FRAC = 1.0
RANDOM_SEED = 42

MAX_LENGTH = 512
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        print("  Using MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("  Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("  Using CPU")
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Generic batch scoring
# ---------------------------------------------------------------------------

def score_batch(texts, tokenizer, model, device, text_pairs=None):
    """Score a batch of texts. Returns softmax probabilities."""
    encodings = tokenizer(
        texts,
        text_pair=text_pairs,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)

    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_bias_detector(df, device):
    """Run bias detection on the sentence column."""
    print("\n" + "=" * 60)
    print("MODEL 1: BIAS DETECTOR (himel7/bias-detector)")
    print("=" * 60)

    print(f"  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BIAS_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BIAS_MODEL)
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    print(f"  Label mapping: {id2label}")

    all_results = []
    total = len(df)
    start_time = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = df.iloc[batch_start:batch_end]
        texts = batch["sentence"].fillna("").tolist()

        probs = score_batch(texts, tokenizer, model, device)

        for i in range(len(texts)):
            scores = probs[i]
            pred_idx = int(scores.argmax())
            pred_label = id2label[pred_idx]
            is_biased = pred_label.lower() in ("biased", "bias", "label_1", "1")

            all_results.append({
                "bias_label": "biased" if is_biased else "non-biased",
                "bias_confidence": float(scores[pred_idx]),
            })

        if (batch_start // BATCH_SIZE) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (batch_start + len(batch)) / elapsed if elapsed > 0 else 0
            remaining = (total - batch_start - len(batch)) / rate if rate > 0 else 0
            print(f"  {batch_start + len(batch)}/{total} "
                  f"({rate:.1f}/sec, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"  Done. {total} sentences in {elapsed:.0f}s ({total/elapsed:.1f}/sec)")

    # Free memory
    del model, tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()

    return pd.DataFrame(all_results)


def run_sentiment(df, device):
    """Run general sentiment on the context column."""
    print("\n" + "=" * 60)
    print("MODEL 2: SENTIMENT (cardiffnlp/twitter-roberta-base-sentiment)")
    print("=" * 60)

    print(f"  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    model.to(device)
    model.eval()

    # Cardiff NLP: 0=negative, 1=neutral, 2=positive
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    all_results = []
    total = len(df)
    start_time = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = df.iloc[batch_start:batch_end]
        texts = batch["context"].fillna("").tolist()

        probs = score_batch(texts, tokenizer, model, device)

        for i in range(len(texts)):
            scores = probs[i]
            pred_idx = int(scores.argmax())
            all_results.append({
                "sentiment": label_map[pred_idx],
                "sent_score_negative": float(scores[0]),
                "sent_score_neutral": float(scores[1]),
                "sent_score_positive": float(scores[2]),
                "sentiment_score": float(scores[2] - scores[0]),
            })

        if (batch_start // BATCH_SIZE) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (batch_start + len(batch)) / elapsed if elapsed > 0 else 0
            remaining = (total - batch_start - len(batch)) / rate if rate > 0 else 0
            print(f"  {batch_start + len(batch)}/{total} "
                  f"({rate:.1f}/sec, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"  Done. {total} contexts in {elapsed:.0f}s ({total/elapsed:.1f}/sec)")

    # Free memory
    del model, tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and sample
    print("Loading mentions...")
    df = pd.read_csv(MENTIONS_PATH)
    df = df[df["party"].isin(TARGET_PARTIES)].copy()
    print(f"  {len(df)} mentions for {TARGET_PARTIES}")

    df_sample = pd.concat([
        group.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED)
        for _, group in df.groupby("party")
    ]).reset_index(drop=True)

    print(f"  Sampled {len(df_sample)} mentions ({SAMPLE_FRAC:.0%})")
    for party in TARGET_PARTIES:
        print(f"    {party}: {(df_sample['party'] == party).sum()}")

    device = get_device()

    # Run both models
    bias_results = run_bias_detector(df_sample, device)
    sent_results = run_sentiment(df_sample, device)

    # Combine everything
    df_out = pd.concat([
        df_sample.reset_index(drop=True),
        bias_results,
        sent_results,
    ], axis=1)

    # Parse dates
    df_out["publish_date"] = pd.to_datetime(df_out["publish_date"],
                                             errors="coerce")
    df_out["year"] = df_out["publish_date"].dt.year
    df_out["is_biased"] = (df_out["bias_label"] == "biased").astype(int)

    # Save
    output_path = os.path.join(OUTPUT_DIR, "analysis_results.csv")
    df_out.to_csv(output_path, index=False)
    print(f"\n  Saved to {output_path}")

    # ===================================================================
    # RESULTS
    # ===================================================================

    print("\n" + "=" * 60)
    print("BIAS DETECTION RESULTS")
    print("=" * 60)

    # Bias rate by party
    print("\n  Bias rate by party:")
    for party in TARGET_PARTIES:
        subset = df_out[df_out["party"] == party]
        n = len(subset)
        n_biased = subset["is_biased"].sum()
        print(f"    {party}: {n_biased/n:.1%} ({n_biased}/{n})")

    # Bias distribution
    print("\n  Bias distribution (%):")
    dist = pd.crosstab(df_out["party"], df_out["bias_label"],
                       normalize="index")
    print((dist * 100).round(1).to_string())

    # Quote vs non-quote
    print("\n  Bias rate by party and quote status:")
    quote_bias = df_out.groupby(
        ["party", "is_quote"])["is_biased"].mean()
    print(quote_bias.round(3).to_string())

    # By year
    print("\n  Bias rate by party and year:")
    yearly_bias = df_out.groupby(
        ["year", "party"])["is_biased"].mean().unstack()
    print((yearly_bias * 100).round(1).to_string())

    # ===================================================================
    # SENTIMENT RESULTS
    # ===================================================================

    print("\n" + "=" * 60)
    print("SENTIMENT RESULTS (Cardiff NLP)")
    print("=" * 60)

    print("\n  Mean sentiment score by party (-1 to +1):")
    sent_means = df_out.groupby("party")["sentiment_score"].agg(
        ["mean", "std", "count"])
    print(sent_means.round(3).to_string())

    print("\n  Sentiment distribution (%):")
    sent_dist = pd.crosstab(df_out["party"], df_out["sentiment"],
                            normalize="index")
    print((sent_dist * 100).round(1).to_string())

    # ===================================================================
    # BIAS DIRECTION (combined)
    # ===================================================================

    print("\n" + "=" * 60)
    print("BIAS DIRECTION (bias detector + sentiment)")
    print("=" * 60)

    median_sent = df_out["sentiment_score"].median()
    print(f"\n  Median sentiment (all sentences): {median_sent:+.3f}")
    print(f"  Sentences below median = relatively negative")

    biased = df_out[df_out["is_biased"] == 1].copy()
    biased["bias_direction"] = "neutral"
    biased.loc[biased["sentiment_score"] < median_sent, "bias_direction"] = "negative"
    biased.loc[biased["sentiment_score"] > median_sent, "bias_direction"] = "positive"

    print(f"\n  Total biased sentences: {len(biased)}")

    print("\n  Bias direction by party (relative to median):")
    dir_dist = pd.crosstab(biased["party"], biased["bias_direction"],
                           normalize="index")
    print((dir_dist * 100).round(1).to_string())

    print("\n  Bias direction counts:")
    dir_counts = pd.crosstab(biased["party"], biased["bias_direction"])
    print(dir_counts.to_string())

    # Mean sentiment of biased vs non-biased
    print("\n  Mean sentiment — biased sentences:")
    print(biased.groupby("party")["sentiment_score"].agg(
        ["mean", "std", "count"]).round(3).to_string())

    non_biased = df_out[df_out["is_biased"] == 0]
    print("\n  Mean sentiment — non-biased sentences:")
    print(non_biased.groupby("party")["sentiment_score"].agg(
        ["mean", "std", "count"]).round(3).to_string())

    # Editorial vs quoted direction
    print("\n  Bias direction by party and quote status:")
    for party in TARGET_PARTIES:
        print(f"\n    {party}:")
        party_biased = biased[biased["party"] == party]
        for is_quote in [False, True]:
            label = "Quoted" if is_quote else "Editorial"
            subset = party_biased[party_biased["is_quote"] == is_quote]
            if len(subset) == 0:
                continue
            neg = (subset["bias_direction"] == "negative").sum()
            pos = (subset["bias_direction"] == "positive").sum()
            total_sub = len(subset)
            print(f"      {label} (n={total_sub}): "
                  f"neg={neg} ({neg/total_sub:.0%}), "
                  f"pos={pos} ({pos/total_sub:.0%})")

    # Yearly negative bias rate
    print("\n  Negative bias rate by party and year:")
    df_out["is_neg_biased"] = (
        (df_out["is_biased"] == 1) &
        (df_out["sentiment_score"] < median_sent)
    ).astype(int)
    yearly_neg = df_out.groupby(
        ["year", "party"])["is_neg_biased"].mean().unstack()
    print((yearly_neg * 100).round(1).to_string())

    # ===================================================================
    # SUMMARY
    # ===================================================================

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for party in TARGET_PARTIES:
        subset = df_out[df_out["party"] == party]
        n = len(subset)
        bias_pct = subset["is_biased"].mean()
        sent_mean = subset["sentiment_score"].mean()

        biased_sub = subset[subset["is_biased"] == 1]
        neg_bias_rate = (biased_sub["sentiment_score"] < median_sent).mean() \
            if len(biased_sub) > 0 else 0
        pos_bias_rate = (biased_sub["sentiment_score"] > median_sent).mean() \
            if len(biased_sub) > 0 else 0

        # Overall negative bias rate (as % of ALL mentions)
        neg_bias_overall = (
            (subset["is_biased"] == 1) &
            (subset["sentiment_score"] < median_sent)
        ).mean()

        print(f"\n  {party} (n={n}):")
        print(f"    Biased language rate:          {bias_pct:.1%}")
        print(f"    General sentiment:             {sent_mean:+.3f}")
        print(f"    Of biased sentences:")
        print(f"      Relatively negative:         {neg_bias_rate:.1%}")
        print(f"      Relatively positive:         {pos_bias_rate:.1%}")
        print(f"    Overall negative bias rate:     {neg_bias_overall:.1%}")
        print(f"      (= biased + below-median sentiment, as % of all mentions)")


if __name__ == "__main__":
    main()