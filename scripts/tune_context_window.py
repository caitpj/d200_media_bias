"""
tune_context_window.py — Experiment with context window sizes for bias
detection and sentiment analysis.

Tests both models at window sizes of 0 (sentence only), 1, and 2
surrounding sentences. Outputs a comparison table and flags cases
where the window size changes the classification.

Usage:
    python scripts/tune_context_window.py

Requirements:
    pip install pandas torch transformers
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
RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                        "nation_cymru_articles.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tuning")

BIAS_MODEL = "himel7/bias-detector"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

WINDOW_SIZES = [0, 1, 2]   # 0 = sentence only, 1 = ±1, 2 = ±2

TARGET_PARTIES = ["reform_uk", "plaid_cymru", "labour", "conservative"]

# Use a manageable sample for the experiment
SAMPLE_N = 500
RANDOM_SEED = 42

MAX_LENGTH = 512
BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re
import json


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting common abbreviations."""
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|Rev)\.',
                  r'\1DOTPH', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.replace('DOTPH', '.') for s in sentences]
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def score_batch(texts, tokenizer, model, device):
    """Score a batch of texts. Returns softmax probabilities."""
    encodings = tokenizer(
        texts,
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
# Build context windows at multiple sizes
# ---------------------------------------------------------------------------

def build_windows(mentions_df: pd.DataFrame, articles: list) -> pd.DataFrame:
    """
    For each mention, reconstruct the context at each window size.
    Returns the mentions df with additional columns:
      context_w0 (sentence only), context_w1 (±1), context_w2 (±2)
    """
    # Build article lookup: url -> text
    article_texts = {a["url"]: a["text"] for a in articles}

    rows = []
    for _, row in mentions_df.iterrows():
        text = article_texts.get(row["url"], "")
        if not text:
            continue

        sentences = split_sentences(text)
        sent_idx = row.get("sentence_idx", None)

        # If we don't have sentence_idx, find it by matching
        if pd.isna(sent_idx):
            target = row["sentence"]
            for i, s in enumerate(sentences):
                if target in s or s in target:
                    sent_idx = i
                    break
            if pd.isna(sent_idx):
                continue
        sent_idx = int(sent_idx)

        windows = {}
        for w in WINDOW_SIZES:
            start = max(0, sent_idx - w)
            end = min(len(sentences), sent_idx + w + 1)
            windows[f"context_w{w}"] = " ".join(sentences[start:end])

        rows.append({**row.to_dict(), **windows})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run a model at each window size
# ---------------------------------------------------------------------------

def run_model_at_windows(df, model_name, input_cols, device, model_type="bias"):
    """
    Run a model on multiple input columns (one per window size).
    Returns a dict of {col_name: results_df}.
    """
    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    results = {}

    for col in input_cols:
        window_label = col.replace("context_w", "w")
        print(f"    Running on {col} ({window_label})...")
        start_time = time.time()

        col_results = []
        texts = df[col].fillna("").tolist()

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(texts))
            batch_texts = texts[batch_start:batch_end]
            probs = score_batch(batch_texts, tokenizer, model, device)

            for i in range(len(batch_texts)):
                scores = probs[i]
                pred_idx = int(scores.argmax())

                if model_type == "bias":
                    id2label = model.config.id2label
                    pred_label = id2label[pred_idx]
                    is_biased = pred_label.lower() in (
                        "biased", "bias", "label_1", "1")
                    col_results.append({
                        f"bias_{window_label}": "biased" if is_biased else "non-biased",
                        f"bias_conf_{window_label}": float(scores[pred_idx]),
                    })
                else:  # sentiment
                    sent_score = float(scores[2] - scores[0])
                    label_map = {0: "negative", 1: "neutral", 2: "positive"}
                    col_results.append({
                        f"sentiment_{window_label}": label_map[pred_idx],
                        f"sent_score_{window_label}": sent_score,
                    })

        elapsed = time.time() - start_time
        print(f"      Done in {elapsed:.1f}s")
        results[col] = pd.DataFrame(col_results)

    # Free memory
    del model, tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # Load mentions
    print("\nLoading mentions...")
    df = pd.read_csv(MENTIONS_PATH)
    df = df[df["party"].isin(TARGET_PARTIES)].copy()

    # Sample
    df_sample = df.groupby("party").apply(
        lambda x: x.sample(min(SAMPLE_N // len(TARGET_PARTIES), len(x)),
                           random_state=RANDOM_SEED)
    ).reset_index(drop=True)
    print(f"  Sampled {len(df_sample)} mentions")
    print(df_sample["party"].value_counts().to_string())

    # Load raw articles for sentence reconstruction
    print("\nLoading raw articles...")
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"  {len(articles)} articles loaded")

    # Build context windows
    print("\nBuilding context windows at sizes:", WINDOW_SIZES)
    df_windows = build_windows(df_sample, articles)
    print(f"  {len(df_windows)} mentions with windows built")

    if len(df_windows) == 0:
        print("\n  ERROR: No mentions matched to articles.")
        print("  Check that URLs in party_mentions.csv match URLs in raw articles.")
        print("  Sample mention URLs:")
        print(df_sample["url"].head().to_string())
        print("  Sample article URLs:")
        print([a["url"] for a in articles[:5]])
        return

    # Verify columns exist
    window_cols = [f"context_w{w}" for w in WINDOW_SIZES]
    for col in window_cols:
        avg_len = df_windows[col].str.len().mean()
        print(f"  {col}: avg {avg_len:.0f} chars")

    # ===================================================================
    # BIAS DETECTION at each window
    # ===================================================================
    print("\n" + "=" * 60)
    print("BIAS DETECTION — varying context window")
    print("=" * 60)

    bias_results = run_model_at_windows(
        df_windows, BIAS_MODEL, window_cols, device, model_type="bias")

    # ===================================================================
    # SENTIMENT at each window
    # ===================================================================
    print("\n" + "=" * 60)
    print("SENTIMENT — varying context window")
    print("=" * 60)

    sent_results = run_model_at_windows(
        df_windows, SENTIMENT_MODEL, window_cols, device, model_type="sentiment")

    # Combine all results — merge each dict separately to avoid key collisions
    combined = df_windows.reset_index(drop=True)
    for res_df in bias_results.values():
        combined = pd.concat([combined, res_df.reset_index(drop=True)], axis=1)
    for res_df in sent_results.values():
        combined = pd.concat([combined, res_df.reset_index(drop=True)], axis=1)

    print(f"\n  Combined columns: {list(combined.columns)}")

    # Save full results
    out_path = os.path.join(OUTPUT_DIR, "context_window_experiment.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # ===================================================================
    # COMPARISON TABLE
    # ===================================================================
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    # --- Bias rates by window size and party ---
    print("\n  Bias rate (%) by party and window size:")
    print(f"  {'Party':<15s}", end="")
    for w in WINDOW_SIZES:
        print(f"  w={w:>3d}  ", end="")
    print()
    print(f"  {'-'*45}")

    for party in TARGET_PARTIES:
        subset = combined[combined["party"] == party]
        print(f"  {party:<15s}", end="")
        for w in WINDOW_SIZES:
            col = f"bias_w{w}"
            rate = (subset[col] == "biased").mean() * 100
            print(f"  {rate:5.1f}%  ", end="")
        print()

    # --- Sentiment scores by window size and party ---
    print("\n  Mean sentiment score by party and window size:")
    print(f"  {'Party':<15s}", end="")
    for w in WINDOW_SIZES:
        print(f"  w={w:>3d}  ", end="")
    print()
    print(f"  {'-'*45}")

    for party in TARGET_PARTIES:
        subset = combined[combined["party"] == party]
        print(f"  {party:<15s}", end="")
        for w in WINDOW_SIZES:
            col = f"sent_score_w{w}"
            mean = subset[col].mean()
            print(f"  {mean:+.3f}  ", end="")
        print()

    # --- Classification stability ---
    print("\n  Classification stability (% mentions that change label):")

    for model_type, prefix in [("Bias", "bias"), ("Sentiment", "sentiment")]:
        w0_col = f"{prefix}_w0"
        w2_col = f"{prefix}_w2"
        if w0_col in combined.columns and w2_col in combined.columns:
            changed = (combined[w0_col] != combined[w2_col]).mean() * 100
            print(f"    {model_type}: {changed:.1f}% of mentions change "
                  f"between w=0 and w=2")

            # Breakdown by party
            for party in TARGET_PARTIES:
                subset = combined[combined["party"] == party]
                party_changed = (subset[w0_col] != subset[w2_col]).mean() * 100
                print(f"      {party}: {party_changed:.1f}%")

    # --- Example flips ---
    print("\n  Example classification flips (w=0 → w=2):")
    w0_col = "bias_w0"
    w2_col = "bias_w2"
    flips = combined[combined[w0_col] != combined[w2_col]].head(5)
    for _, row in flips.iterrows():
        print(f"\n    Party: {row['party']}")
        print(f"    Sentence: {row['sentence'][:100]}...")
        print(f"    w=0: {row[w0_col]}  →  w=2: {row[w2_col]}")
        print(f"    Context (w=2): {row['context_w2'][:150]}...")

    print(f"\n  Full results saved to: {out_path}")
    print("  Use these to select your preferred window size.")


if __name__ == "__main__":
    main()