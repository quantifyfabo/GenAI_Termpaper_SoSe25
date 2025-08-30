#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 21:37:30 2025

@author: fabiangi
"""

# final?

# -*- coding: utf-8 -*-
"""
LLM-based topic classification using Ollama (Llama 3.1 8B), full dataset.
- Reads input CSV
- Uses first 500 words from 'plain_text_hard'
- Strict zero-shot prompt: exactly one of 4 labels
- JSON mode first, fallback to plain text
- Normalizes model output to a valid label when possible; else keeps cleaned raw text
- Writes predictions to 'llm_topic_name' and saves augmented CSV
- Deterministic settings (seed=2025)
"""

import re
import json
import time
import pandas as pd
from tqdm import tqdm
import ollama

# -------- Paths --------
INPUT_PATH  = "/Users/fabiangi/Documents/Goethe Uni/Uni_Projects/CSS_Gen_AI/Paper_GenAI_SoSe25/Data/Zwischen_Datasets/NYT_Dataset_w_Topics_v2.csv"
OUTPUT_PATH = "/Users/fabiangi/Documents/Goethe Uni/Uni_Projects/CSS_Gen_AI/Paper_GenAI_SoSe25/Data/Zwischen_Datasets/NYT_Dataset_w_Topics_v2_with_LLM.csv"

# -------- Labels (exact casing) --------
LABELS = [
    "Climate Change",
    "Migration Policy",
    "Economic Inequality",
    "Digital Surveillance",
]

# -------- LLM configuration --------
MODEL_NAME     = "llama3.1:8b"
TEMPERATURE    = 0.0
TOP_P          = 1.0
SEED           = 2025     # deterministic
NUM_PREDICT    = 32       # small, we expect a single label (JSON text adds a few chars)
RETRIES        = 2        # retry on transient errors

# -------- Truncation --------
TEXT_COLUMN    = "plain_text_hard"
TRUNC_WORDS    = 500


# =========================
# Helper functions
# =========================
def truncate_to_n_words(text: str, n: int = TRUNC_WORDS) -> str:
    """Return the first n words of text (safe for non-str)."""
    if not isinstance(text, str):
        return ""
    words = text.strip().split()
    return " ".join(words[:n])


def system_prompt_json() -> str:
    """Strict JSON-only instruction."""
    return (
        "Output a JSON object with a single key 'label'. "
        "The value must be exactly one of: "
        f"{', '.join(LABELS)}. "
        "No explanations. No additional keys."
    )


def user_prompt(document_text: str) -> str:
    """User prompt containing the text to classify."""
    return (
        "Classify the following news text into exactly one of the specified labels.\n\n"
        f"{document_text}"
    )


def extract_resp_content(resp) -> str:
    """Support both ChatResponse objects and dict-like responses."""
    # ChatResponse (modern ollama)
    if hasattr(resp, "message") and hasattr(resp.message, "content"):
        return resp.message.content or ""
    # Dict-like fallback
    if isinstance(resp, dict):
        msg = resp.get("message", {})
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
    # Last resort: introspect __dict__
    try:
        d = getattr(resp, "__dict__", {})
        msg = d.get("message", {})
        if hasattr(msg, "content"):
            return msg.content or ""
        if isinstance(msg, dict):
            return msg.get("content", "") or ""
    except Exception:
        pass
    return ""


def normalize_label_or_keep(raw: str) -> str:
    """
    Map output to an allowed label if possible; otherwise return cleaned raw.
    Accepts exact match, startswith, or word-boundary contains.
    """
    if raw is None:
        return "RAW_EMPTY"

    t = raw.strip()
    # remove common prefixes like "Label:", "Category:", "Answer:"
    t = re.sub(r'^(label|category|answer)\s*:\s*', '', t, flags=re.IGNORECASE).strip()
    # strip frequent surrounding punctuation
    t = t.strip(' \n\r\t"\'`*-_.,;:()[]{}')

    # exact (case-insensitive)
    for lab in LABELS:
        if t.lower() == lab.lower():
            return lab

    # startswith (e.g., "Climate Change." or "Climate Change article")
    for lab in LABELS:
        if t.lower().startswith(lab.lower()):
            return lab

    # contains with word boundary
    for lab in LABELS:
        if re.search(rf'\b{re.escape(lab)}\b', t, flags=re.IGNORECASE):
            return lab

    # no match: return cleaned raw to allow manual inspection
    return t if t else "RAW_EMPTY"


def classify_one_json_first(doc_text: str) -> str:
    """
    Try JSON mode first (format='json'), then fallback to plain-text.
    Always return a non-empty string (normalized label or cleaned raw).
    """
    sys = system_prompt_json()
    usr = user_prompt(doc_text)

    # 1) JSON mode
    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            resp = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user",   "content": usr},
                ],
                options={"temperature": TEMPERATURE, "top_p": TOP_P, "seed": SEED, "num_predict": NUM_PREDICT},
                format="json",  # enforce JSON
            )
            raw = extract_resp_content(resp)

            # try to parse strict JSON
            try:
                obj = json.loads(raw)
                label_val = obj.get("label", "")
                if isinstance(label_val, str) and label_val.strip():
                    return normalize_label_or_keep(label_val)
            except Exception:
                # fall through to normalization over raw
                pass

            return normalize_label_or_keep(raw)

        except Exception as e:
            last_err = e
            time.sleep(0.3)
            continue

    # 2) Fallback plain text (no format arg)
    try:
        resp2 = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content":
                    "You must answer with exactly one of the following labels, and nothing else:\n"
                    + "\n".join(LABELS) +
                    "\nNo explanations. No punctuation. No quotes."
                },
                {"role": "user",   "content": usr},
            ],
            options={"temperature": TEMPERATURE, "top_p": TOP_P, "seed": SEED, "num_predict": 16},
        )
        raw2 = extract_resp_content(resp2)
        return normalize_label_or_keep(raw2)
    except Exception:
        # final fallback
        return f"RAW_ERROR: {last_err}" if last_err else "RAW_ERROR"


# =========================
# Main pipeline
# =========================
# 1) Load data
df = pd.read_csv(INPUT_PATH)
if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Text column '{TEXT_COLUMN}' not found. Available: {list(df.columns)[:20]}")

# 2) Classify with progress bar
preds = []
for i in tqdm(range(len(df)), desc="Classifying with Llama 3.1 8B (Ollama)"):
    raw_text = df.at[i, TEXT_COLUMN]
    doc = truncate_to_n_words(raw_text, TRUNC_WORDS)
    pred = classify_one_json_first(doc)
    preds.append(pred)

# 3) Attach predictions and save
df["llm_topic_name"] = preds

print("\nLLM predicted label counts:")
print(df["llm_topic_name"].value_counts(dropna=False))

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved with LLM labels → {OUTPUT_PATH}")
