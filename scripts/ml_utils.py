"""
ml_utils.py — Shared model loading, scoring, and preprocessing utilities.

Two-stage pipeline:
  1. Bias detection — is the language biased? (himel7/bias-detector)
  2. LLM classification — entity-level sentiment attribution via Claude
     Haiku 4.5 (Anthropic API)

Used by analyse.py (primary) and analyse_secondary.py.
"""

import os
import time
import json

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIAS_MODEL = "himel7/bias-detector"
LLM_MODEL = "claude-haiku-4-5-20251001"
MAX_LENGTH = 512
BATCH_SIZE = 16

# How many mentions to send per API call
LLM_BATCH_SIZE = 100

LLM_SYSTEM_PROMPT = """You are classifying media bias in news articles about Welsh political parties.

For each row, determine:

1. on_target (1 or 0): Is the bias ABOUT the named party?
   - 1 = The text portrays the party positively or negatively (the party is the subject being judged)
   - 0 = The party is merely speaking, being quoted, or mentioned in passing — the bias is about something/someone else
   
   Examples:
   - "Plaid Cymru criticised the devastating policy" → 0 (Plaid is the SPEAKER)
   - "Plaid Cymru set out clear proposals but Labour ignored them" → 0 (negativity targets Labour)
   - "Reform UK's dangerous populist agenda" → 1 (Reform IS the target)
   - "Plaid Cymru's unrealistic independence plans" → 1 (Plaid IS the target)

2. sentiment_score (number from -1.0 to +1.0): If on_target = 1, how is the party portrayed?
   - -1.0 = Strongly negative
   - -0.5 = Moderately negative
   -  0.0 = Neutral or off-target
   - +0.5 = Moderately positive
   - +1.0 = Strongly positive

3. reasoning: One short sentence explaining your decision.

RULES:
- CRITICAL: Assess bias ONLY toward the party named in "ASSESS BIAS TOWARD". If the text praises Party A while criticising Party B, and you are assessing Party B, that is negative toward Party B (on_target=1, negative score). Do not describe how other parties are portrayed.
- Use the "context" for your assessment, not just the "sentence"
- Quoted speech (is_quote = True) usually means the party is speaking — likely on_target = 0
- If the party is criticising something else, that is NOT on_target
- Before returning, review each classification: if your reasoning says the party is NOT being judged, on_target must be 0 and sentiment_score must be 0.0. Fix any contradictions.

Respond ONLY with a JSON array. No other text. Example:
[{"id": 0, "on_target": 1, "sentiment_score": -0.5, "reasoning": "Reform portrayed as ineffective"},{"id": 1, "on_target": 0, "sentiment_score": 0.0, "reasoning": "Plaid is the speaker, not the target"}]"""


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
# Article type from URL
# ---------------------------------------------------------------------------

def get_article_type(url):
    url = str(url).lower()
    if "/opinion/" in url:
        return "opinion"
    elif "/news/" in url:
        return "news"
    return "other"


# ---------------------------------------------------------------------------
# Batch scoring (bias detector)
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
# Stage 1: Bias detection
# ---------------------------------------------------------------------------

def run_bias_detector(df, device):
    """Classify each mention's context as biased or non-biased."""
    print("\n" + "=" * 60)
    print("STAGE 1: BIAS DETECTOR (himel7/bias-detector)")
    print(f"  Running on {len(df)} mentions")
    print("=" * 60)

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
        texts = batch["context"].fillna("").tolist()

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
    print(f"  Done. {total} mentions in {elapsed:.0f}s ({total/elapsed:.1f}/sec)")

    del model, tokenizer
    if device.type == "mps":
        torch.mps.empty_cache()

    return pd.DataFrame(all_results)


# ---------------------------------------------------------------------------
# Stage 1 pipeline (bias detection only)
# ---------------------------------------------------------------------------

def run_bias_pipeline(df, device):
    """
    Run Stage 1 (bias detection) and return results.
    Returns (df_all, df_biased_raw).
    """
    bias_results = run_bias_detector(df, device)

    df_all = pd.concat([
        df.reset_index(drop=True),
        bias_results,
    ], axis=1)

    df_all["publish_date"] = pd.to_datetime(df_all["publish_date"],
                                             errors="coerce")
    df_all["year"] = df_all["publish_date"].dt.year
    df_all["is_biased"] = (df_all["bias_label"] == "biased").astype(int)
    df_all["article_type"] = df_all["url"].apply(get_article_type)

    df_biased = df_all[df_all["is_biased"] == 1].copy()
    print(f"\n  {len(df_biased)} biased mentions "
          f"({len(df_biased)/len(df_all):.1%} of total)")

    return df_all, df_biased


# ---------------------------------------------------------------------------
# Stage 2: LLM classification via Anthropic API
# ---------------------------------------------------------------------------

def run_llm_classification(df_biased, output_dir):
    """
    Classify biased mentions using Claude Haiku 4.5 via the Anthropic API.
    Processes in batches, saves intermediate results, and handles retries.

    Returns DataFrame with on_target, sentiment_score, reasoning columns.
    """
    if not HAS_ANTHROPIC:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Export it: "
            "export ANTHROPIC_API_KEY='your-key-here'"
        )

    print("\n" + "=" * 60)
    print(f"STAGE 2: LLM CLASSIFICATION ({LLM_MODEL})")
    print(f"  {len(df_biased)} biased mentions to classify")
    print(f"  Batch size: {LLM_BATCH_SIZE}")
    print("=" * 60)

    client = anthropic.Anthropic(api_key=api_key)

    # Prepare rows with IDs
    df_biased = df_biased.reset_index(drop=True)
    df_biased["id"] = range(len(df_biased))

    # Check for existing partial results
    partial_path = os.path.join(output_dir, "llm_partial_results.json")
    completed = {}
    if os.path.exists(partial_path):
        with open(partial_path, "r") as f:
            completed = {r["id"]: r for r in json.load(f)}
        print(f"  Resuming: {len(completed)} already classified")

    all_results = list(completed.values())
    done_ids = set(completed.keys())

    # Filter to remaining rows
    remaining = df_biased[~df_biased["id"].isin(done_ids)]
    n_batches = (len(remaining) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
    print(f"  {len(remaining)} remaining, {n_batches} batches")

    start_time = time.time()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * LLM_BATCH_SIZE
        batch_end = min(batch_start + LLM_BATCH_SIZE, len(remaining))
        batch = remaining.iloc[batch_start:batch_end]

        # Build the user message with the batch data
        rows_text = []
        for _, row in batch.iterrows():
            rows_text.append(
                f"id: {row['id']}\n"
                f"ASSESS BIAS TOWARD: {row['party']}\n"
                f"is_quote: {row['is_quote']}\n"
                f"sentence: {row['sentence']}\n"
                f"context: {row['context']}"
            )
        user_msg = ("Classify each mention. For each row, assess bias "
                    "ONLY toward the party named in 'ASSESS BIAS TOWARD'. "
                    "Ignore how other parties are portrayed. "
                    "Respond with ONLY a JSON array.\n\n"
                    + "\n---\n".join(rows_text))

        # API call with retry
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=LLM_MODEL,
                    max_tokens=16384,
                    system=LLM_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    Retry {attempt+1} after error: {e}")
                    time.sleep(wait)
                else:
                    print(f"    FAILED batch {batch_idx+1}: {e}")
                    continue

        # Parse response
        response_text = response.content[0].text.strip()
        # Clean potential markdown wrapping
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rsplit("```", 1)[0]

        try:
            batch_results = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to salvage truncated JSON by finding last complete object
            last_brace = response_text.rfind("}")
            if last_brace > 0:
                truncated = response_text[:last_brace + 1] + "]"
                try:
                    batch_results = json.loads(truncated)
                    print(f"    Recovered {len(batch_results)} from truncated "
                          f"batch {batch_idx+1}")
                except json.JSONDecodeError:
                    print(f"    WARNING: Failed to parse batch {batch_idx+1}")
                    print(f"    First 200 chars: {response_text[:200]}")
                    continue
            else:
                print(f"    WARNING: Failed to parse batch {batch_idx+1}")
                print(f"    First 200 chars: {response_text[:200]}")
                continue

        all_results.extend(batch_results)

        # Save partial results
        with open(partial_path, "w") as f:
            json.dump(all_results, f)

        elapsed = time.time() - start_time
        rate = (batch_end) / elapsed if elapsed > 0 else 0
        remaining_time = (len(remaining) - batch_end) / rate if rate > 0 else 0
        print(f"  Batch {batch_idx+1}/{n_batches}: "
              f"{len(batch_results)} classified "
              f"(~{remaining_time:.0f}s remaining)")

    elapsed = time.time() - start_time
    print(f"\n  Done. {len(all_results)} classifications in {elapsed:.0f}s")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df["id"] = results_df["id"].astype(int)
    results_df["on_target"] = results_df["on_target"].astype(int)
    results_df["sentiment_score"] = results_df["sentiment_score"].astype(float)
    if "reasoning" not in results_df.columns:
        results_df["reasoning"] = ""

    # Save final results
    results_path = os.path.join(output_dir, "llm_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")

    # Clean up partial
    if os.path.exists(partial_path):
        os.remove(partial_path)

    n_on = (results_df["on_target"] == 1).sum()
    n_off = (results_df["on_target"] == 0).sum()
    print(f"  On-target: {n_on} ({n_on/len(results_df):.1%})")
    print(f"  Off-target: {n_off} ({n_off/len(results_df):.1%})")

    return results_df


def merge_llm_results(df_biased, results_df):
    """Merge LLM classification results with biased mentions DataFrame."""
    df_biased = df_biased.reset_index(drop=True)
    df_biased["id"] = range(len(df_biased))

    df_merged = df_biased.merge(
        results_df[["id", "on_target", "sentiment_score", "reasoning"]],
        on="id", how="left",
    )

    n_on_target = (df_merged["on_target"] == 1).sum()
    n_off_target = (df_merged["on_target"] == 0).sum()
    n_missing = df_merged["on_target"].isna().sum()

    print(f"  On-target: {n_on_target}")
    print(f"  Off-target (filtered): {n_off_target}")
    if n_missing > 0:
        print(f"  WARNING: {n_missing} mentions without LLM classification")

    df_merged["weighted_bias_score"] = (
        df_merged["bias_confidence"] * df_merged["sentiment_score"]
    )

    return df_merged