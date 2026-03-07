"""
ml_utils.py — Shared model loading, scoring, and preprocessing utilities.

Two-stage pipeline:
  1. Bias detection — is the language biased? (himel7/bias-detector)
  2. LLM classification — entity-level sentiment attribution via Claude
     Sonnet 4.6 (Anthropic API)

Used by analyse.py (primary) and analyse_secondary.py.
"""

import os
import time
import json

import pandas as pd
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
LLM_MODEL = "claude-sonnet-4-6"
MAX_LENGTH = 512
BATCH_SIZE = 16

# How many mentions to send per API call
LLM_BATCH_SIZE = 50

LLM_SYSTEM_PROMPT = """You are a media bias annotator. For each row, assess whether the text evaluates or judges the party named in "ASSESS BIAS TOWARD".

STEP 1 — TARGET ATTRIBUTION (on_target):
Does the text evaluate, judge, or characterise the named party?
  1 = The party's actions, policies, character, or competence are being assessed.
  0 = The party is merely speaking, being quoted, mentioned in passing, or the judgment targets someone/something else.

STEP 2 — SENTIMENT (sentiment_score):
If on_target = 1, how is the party portrayed? Use ONLY one of these five values:
  -1.0 = Strongly negative (e.g. party described as dangerous, extremist, incompetent)
  -0.5 = Moderately negative (e.g. party criticised, framed unfavourably, failed)
   0.0 = Neutral
  +0.5 = Moderately positive (e.g. party praised, framed favourably, succeeded)
  +1.0 = Strongly positive (e.g. party described as visionary, highly effective)
If on_target = 0, sentiment_score must be 0.0.

EXAMPLES:
- ASSESS: Plaid | "Plaid Cymru criticised the devastating policy" → on_target=0 (Plaid is speaking, not being judged)
- ASSESS: Reform | "Reform UK's dangerous populist agenda threatens Wales" → on_target=1, sentiment_score=-1.0 (Reform characterised as dangerous)
- ASSESS: Plaid | "Plaid Cymru's unrealistic independence plans" → on_target=1, sentiment_score=-0.5 (Plaid's policy judged negatively)
- ASSESS: Reform | "Reform UK failed to win any seats in the region" → on_target=1, sentiment_score=-0.5 (negative factual framing)
- ASSESS: Plaid | "Experts praised Plaid's innovative housing strategy" → on_target=1, sentiment_score=+0.5 (positive third-party assessment)
- ASSESS: Reform | "Labour attacked Reform's record on healthcare" → on_target=1, sentiment_score=-0.5 (Reform's record judged negatively)
- ASSESS: Labour | "Labour attacked Reform's record on healthcare" → on_target=0 (Labour is the speaker, not being judged)
- ASSESS: Plaid | "Plaid set out clear proposals but Labour ignored them" → on_target=0 (negativity targets Labour, not Plaid)

RULES:
- Assess ONLY the party named in "ASSESS BIAS TOWARD". If text criticises Party A while praising Party B, and you are assessing Party A, that is negative toward Party A.
- Use the full "context" field, not just "sentence".
- Write your reasoning BEFORE deciding on_target and sentiment_score. If your reasoning concludes the party is not being judged, on_target must be 0.

OUTPUT: JSON array only. No other text. Put reasoning first in each object.
[{"id": 0, "reasoning": "Reform characterised as dangerous", "on_target": 1, "sentiment_score": -1.0}, {"id": 1, "reasoning": "Plaid is speaking, not being judged", "on_target": 0, "sentiment_score": 0.0}]"""


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
# Stage 1: Bias detection
# ---------------------------------------------------------------------------

def run_bias_detector(df, device):
    """Classify each mention's context as biased or non-biased."""
    print("\n" + "=" * 60)
    print("STAGE 1: BIAS DETECTOR (himel7/bias-detector)")
    print(f"  Running on {len(df)} mentions")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
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
            # Model card: LABEL_1 = biased, LABEL_0 = non-biased
            is_biased = pred_idx == 1

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
    Returns (df_biased, counts) where counts is a dict of per-party totals.
    """
    bias_results = run_bias_detector(df, device)

    df_all = pd.concat([
        df.reset_index(drop=True),
        bias_results,
    ], axis=1)

    df_all["is_biased"] = (df_all["bias_label"] == "biased").astype(int)

    # Build counts dict for coverage bias / z-test denominators
    counts = {}
    for party in df_all["party"].unique():
        subset = df_all[df_all["party"] == party]
        counts[party] = {
            "mentions": len(subset),
            "articles": subset["url"].nunique(),
            "biased": int(subset["is_biased"].sum()),
        }

    df_biased = df_all[df_all["is_biased"] == 1].copy()
    print(f"\n  {len(df_biased)} biased mentions "
          f"({len(df_biased)/len(df_all):.1%} of total)")

    return df_biased, counts


# ---------------------------------------------------------------------------
# Stage 2: LLM classification via Anthropic API
# ---------------------------------------------------------------------------

def run_llm_classification(df_biased, output_dir,
                           results_filename="stage2_llm_raw.tsv",
                           partial_filename="stage2_llm_partial.json"):
    """
    Classify biased mentions using Claude Sonnet 4.6 via the Anthropic API.
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
    partial_path = os.path.join(output_dir, partial_filename)
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

    def _call_and_parse(batch_df):
        """Send a batch to the API and parse. Returns list of results or None."""
        rows_text = []
        for _, row in batch_df.iterrows():
            rows_text.append(
                f"id: {row['id']}\n"
                f"ASSESS BIAS TOWARD: {row['party']}\n"
                f"sentence: {row['sentence']}\n"
                f"context: {row['context']}"
            )
        user_msg = ("Classify each mention. For each row, assess bias "
                    "ONLY toward the party named in 'ASSESS BIAS TOWARD'. "
                    "Ignore how other parties are portrayed. "
                    "Respond with ONLY a JSON array.\n\n"
                    + "\n---\n".join(rows_text))

        response = None
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=LLM_MODEL,
                    max_tokens=16384,
                    system=LLM_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                    thinking={"type": "disabled"},
                )
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    Retry {attempt+1} after error: {e}")
                    time.sleep(wait)
                else:
                    print(f"    FAILED API call: {e}")

        if response is None:
            return None

        response_text = response.content[0].text.strip()

        if response.stop_reason == "max_tokens":
            print(f"    Truncated (hit max_tokens), {len(batch_df)} rows")

        # Clean potential markdown wrapping
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rsplit("```", 1)[0]

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to salvage truncated JSON
            search_from = len(response_text) - 1
            for _ in range(50):
                pos = response_text.rfind("}", 0, search_from)
                if pos <= 0:
                    break
                candidate = response_text[:pos + 1] + "]"
                try:
                    results = json.loads(candidate)
                    print(f"    Recovered {len(results)} from truncated batch")
                    return results
                except json.JSONDecodeError:
                    search_from = pos
            return None

    def _classify_with_retry(batch_df, depth=0):
        """Classify a batch; on failure, split in half and retry."""
        results = _call_and_parse(batch_df)
        if results is not None:
            return results

        # Give up on single rows
        if len(batch_df) <= 1:
            print(f"    Giving up on id={batch_df.iloc[0]['id']}")
            return []

        # Split in half and retry
        mid = len(batch_df) // 2
        print(f"    Split failed batch ({len(batch_df)} rows) "
              f"into {mid} + {len(batch_df) - mid}")
        left = _classify_with_retry(batch_df.iloc[:mid], depth + 1)
        right = _classify_with_retry(batch_df.iloc[mid:], depth + 1)
        return left + right

    for batch_idx in range(n_batches):
        batch_start = batch_idx * LLM_BATCH_SIZE
        batch_end = min(batch_start + LLM_BATCH_SIZE, len(remaining))
        batch = remaining.iloc[batch_start:batch_end]

        batch_results = _classify_with_retry(batch)
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

    # Check for missing IDs and retry them
    classified_ids = {r["id"] for r in all_results}
    expected_ids = set(df_biased["id"].tolist())
    missing_ids = expected_ids - classified_ids
    if missing_ids:
        print(f"\n  Retrying {len(missing_ids)} missing IDs...")
        missing_df = df_biased[df_biased["id"].isin(missing_ids)]
        retry_results = _classify_with_retry(missing_df)
        all_results.extend(retry_results)
        with open(partial_path, "w") as f:
            json.dump(all_results, f)
        still_missing = expected_ids - {r["id"] for r in all_results}
        if still_missing:
            print(f"  WARNING: {len(still_missing)} IDs still missing "
                  f"after retry: {sorted(still_missing)[:20]}")
        else:
            print(f"  All {len(expected_ids)} IDs classified.")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df["id"] = results_df["id"].astype(int)
    results_df["on_target"] = results_df["on_target"].astype(int)
    results_df["sentiment_score"] = results_df["sentiment_score"].astype(float)
    if "reasoning" not in results_df.columns:
        results_df["reasoning"] = ""

    # Save final results
    results_path = os.path.join(output_dir, results_filename)
    results_df.to_csv(results_path, index=False, sep="\t")
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

    return df_merged