# evaluation/gpt_pipeline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from configs.config_loader import load_config

config = load_config()

# ── OpenAI client ─────────────────────────────────────────────────────────────

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except ImportError:
    raise ImportError("Run: pip install openai")

# ── Sentence embeddings for retrieval ────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except ImportError:
    raise ImportError("Run: pip install sentence-transformers")

LABEL_NAMES = ["normal", "offensive", "hatespeech"]
MODEL_NAME  = "gpt-4.1-mini"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """Load train and test splits with rationale masks."""
    data_dir = Path(config["paths"]["data"])

    with open(data_dir / "train.json", encoding="utf-8") as f:
        train = json.load(f)
    with open(data_dir / "test.json", encoding="utf-8") as f:
        test = json.load(f)

    return train, test


def load_cf_lookup():
    """Build lookup: original_id -> counterfactual_text."""
    cf_path = Path(config["paths"]["cf_pairs"]) / "train_cf_pairs.jsonl"
    lookup  = {}

    if not cf_path.exists():
        print(f"Warning: CF pairs not found at {cf_path}")
        return lookup

    with open(cf_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            orig_id = pair["original"]["id"]
            cf_text = pair["counterfactual"]["text"]
            lookup[orig_id] = cf_text

    print(f"Loaded {len(lookup)} CF pairs")
    return lookup


def get_rationale_tokens(example):
    """Extract human-annotated harmful tokens from rationale mask."""
    text   = example.get("text", "")
    tokens = text.split()
    mask   = example.get("rationale_mask", [])

    if not mask or not any(mask):
        return []

    rationale_tokens = [
        tokens[i] for i, val in enumerate(mask)
        if val == 1 and i < len(tokens)
    ]
    return rationale_tokens


# ── Embedding index ───────────────────────────────────────────────────────────

def build_train_index(train_examples):
    """Embed all training examples for cosine similarity retrieval."""
    print("Building embedding index for train examples...")
    texts      = [ex["text"] for ex in train_examples]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print(f"Built index for {len(train_examples)} examples")
    return embeddings


def retrieve_similar(
    query_text:        str,
    train_examples:    list,
    train_embeddings:  np.ndarray,
    cf_lookup:         dict,
    k:                 int = 3,
    ensure_all_labels: bool = True,
) -> list:
    """
    Retrieve one example per class — the most similar example
    from each of normal, offensive, and hatespeech classes.
    This ensures the model always sees all three labels in context.
    """
    query_emb = embedder.encode([query_text])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    scores    = (train_embeddings @ query_emb.T).squeeze()

    # Get best example per class
    selected = []
    for target_label in ["normal", "offensive", "hatespeech"]:
        # Find indices belonging to this class
        class_indices = [
            i for i, ex in enumerate(train_examples)
            if ex.get("label") == target_label
        ]
        if not class_indices:
            continue
        # Get scores for this class only
        class_scores = [(scores[i], i) for i in class_indices]
        best_idx = max(class_scores, key=lambda x: x[0])[1]
        selected.append(train_examples[best_idx])

    return selected


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_zero_shot_prompt(text: str) -> str:
    return (
        "You are an expert hate speech classifier. "
        "Classify the following text into exactly one category:\n\n"
        "- normal: harmless text with no offensive or hateful content\n"
        "- offensive: rude, vulgar, or disrespectful text but NOT targeting "
        "a specific group based on race, religion, gender, or ethnicity\n"
        "- hatespeech: text that attacks or demeans a group based on "
        "protected characteristics like race, religion, gender, or ethnicity\n\n"
        f"Text: {text}\n\n"
        "Respond with exactly one word: normal, offensive, or hatespeech."
    )


def build_few_shot_prompt(
    text:             str,
    similar_examples: list,
    cf_lookup:        dict,
) -> str:
    examples_str = ""
    for i, ex in enumerate(similar_examples, 1):
        examples_str += (
            f"Example {i}:\n"
            f"Text: {ex['text']}\n"
            f"Label: {ex['label']}\n\n"
        )

    return (
        "You are an expert hate speech classifier. "
        "Classify text into exactly one category:\n\n"
        "- normal: harmless text with no offensive or hateful content\n"
        "- offensive: rude or disrespectful text but NOT targeting a specific "
        "group based on race, religion, gender, or ethnicity\n"
        "- hatespeech: text that attacks or demeans a group based on protected "
        "characteristics like race, religion, gender, or ethnicity\n\n"
        "Here are similar examples:\n\n"
        f"{examples_str}"
        "Now classify:\n"
        f"Text: {text}\n\n"
        "Respond with exactly one word: normal, offensive, or hatespeech."
    )


def build_proposed_prompt(
    text:             str,
    similar_examples: list,
    cf_lookup:        dict,
) -> str:
    examples_str = ""
    for i, ex in enumerate(similar_examples, 1):
        ex_id      = ex.get("id", "")
        cf_text    = cf_lookup.get(ex_id, None)
        rat_tokens = get_rationale_tokens(ex)
        label      = ex.get("label", "normal")

        examples_str += f"Example {i}:\n"
        examples_str += f"Text: {ex['text']}\n"

        if rat_tokens:
            examples_str += f"Key harmful phrases: {rat_tokens}\n"

        if cf_text and label != "normal":
            examples_str += (
                f"Neutral rewrite: {cf_text}\n"
                f"(Note: the neutral rewrite removes the harmful intent "
                f"while keeping the same topic)\n"
            )

        examples_str += f"Label: {label}\n\n"

    return (
        "You are an expert hate speech classifier. "
        "Classify text into exactly one category:\n\n"
        "- normal: harmless text with no offensive or hateful content\n"
        "- offensive: rude or disrespectful text but NOT targeting a specific "
        "group based on race, religion, gender, or ethnicity\n"
        "- hatespeech: text that attacks or demeans a group based on protected "
        "characteristics like race, religion, gender, or ethnicity\n\n"
        "To help you classify, each example below includes two types of annotations:\n"
        "1. Key harmful phrases: the specific words or phrases that human annotators "
        "identified as carrying the harmful intent in that text. If a text has "
        "similar phrases, it is likely harmful.\n"
        "2. Neutral rewrite: a minimally edited version of the harmful text that "
        "removes the harmful intent while keeping the same topic. Comparing the "
        "original with its neutral rewrite reveals exactly what makes the original "
        "harmful — the difference is the harmful element.\n\n"
        "Here are similar examples with these annotations:\n\n"
        f"{examples_str}"
        "Now classify this text:\n"
        f"Text: {text}\n\n"
        "Step 1: Does this text target a group based on protected characteristics "
        "(race, religion, gender, ethnicity)? If yes, lean toward hatespeech.\n"
        "Step 2: Look at the key harmful phrases in the examples above — does this "
        "text contain similar language patterns?\n"
        "Step 3: Compare this text with the neutral rewrites above — what would "
        "need to change to make this text neutral? If very little needs to change, "
        "the harm is subtle (likely hatespeech). If it is just rude language with "
        "no group targeting, it is offensive.\n\n"
        "Respond with exactly one word: normal, offensive, or hatespeech."
    )


# ── GPT API call ──────────────────────────────────────────────────────────────

def classify_with_gpt(
    prompt:     str,
    model:      str = MODEL_NAME,
    max_retries: int = 3,
) -> str:
    """Call GPT API and return predicted label."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower()

            # Parse response to one of three labels
            raw_clean = raw.replace("-", "").replace("_", "").strip()

            if "hatespeech" in raw_clean or "hate speech" in raw_clean:
                return "hatespeech"
            elif "offensive" in raw_clean:
                return "offensive"
            elif "normal" in raw_clean:
                return "normal"
            else:
                # Try partial matches
                if "hate" in raw_clean:
                    return "hatespeech"
                elif "offens" in raw_clean:
                    return "offensive"
                else:
                    print(f"  Unclear response: '{raw}' — defaulting to normal")
                    return "normal"

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  API error (attempt {attempt+1}): {e} — retrying in 5s")
                time.sleep(5)
            else:
                print(f"  API error after {max_retries} attempts: {e}")
                return "normal"


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_metrics(true_labels: list, pred_labels: list) -> dict:
    """Compute all metrics matching your existing evaluation."""
    label2id = {"normal": 0, "offensive": 1, "hatespeech": 2}

    y_true = [label2id[l] for l in true_labels]
    y_pred = [label2id[l] for l in pred_labels]

    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class   = f1_score(y_true, y_pred, average=None,       zero_division=0)
    accuracy    = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Harmful-subset F1 (offensive + hatespeech only)
    harmful_mask   = [i for i, l in enumerate(y_true) if l > 0]
    harmful_true   = [y_true[i] for i in harmful_mask]
    harmful_pred   = [y_pred[i] for i in harmful_mask]
    harmful_f1     = f1_score(harmful_true, harmful_pred,
                              average="macro", zero_division=0)

    # Binary F1 (normal vs harmful)
    binary_true = [0 if l == 0 else 1 for l in y_true]
    binary_pred = [0 if l == 0 else 1 for l in y_pred]
    binary_f1   = f1_score(binary_true, binary_pred,
                           average="binary", zero_division=0)

    return {
        "accuracy":          accuracy,
        "macro_f1":          macro_f1,
        "weighted_f1":       weighted_f1,
        "normal_f1":         per_class[0],
        "offensive_f1":      per_class[1],
        "hatespeech_f1":     per_class[2],
        "harmful_subset_f1": harmful_f1,
        "binary_f1":         binary_f1,
        "confusion_matrix":  conf_matrix.tolist(),
    }


def print_results(name: str, metrics: dict):
    """Print formatted results matching your existing evaluation output."""
    print(f"\n{'='*55}")
    print(f"  {name.upper()} — Evaluation Results")
    print(f"{'='*55}")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:         {metrics['weighted_f1']:.4f}")
    print(f"{'─'*55}")
    print(f"  Per-class F1:")
    print(f"    Normal:            {metrics['normal_f1']:.4f}")
    print(f"    Offensive:         {metrics['offensive_f1']:.4f}")
    print(f"    Hate Speech:       {metrics['hatespeech_f1']:.4f}")
    print(f"{'─'*55}")
    print(f"  Harmful-subset F1:   {metrics['harmful_subset_f1']:.4f}  ← KEY METRIC")
    print(f"  Binary F1:           {metrics['binary_f1']:.4f}")
    print(f"{'='*55}\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    mode:       str  = "all",
    max_test:   int  = None,
    save_results: bool = True,
):
    """
    Run the GPT classification pipeline.

    Args:
        mode:        "zero_shot" | "few_shot" | "proposed" | "all"
        max_test:    limit test examples (for quick testing, set to 50)
        save_results: save results to JSON
    """
    print("=== GPT-4.1-mini Agentic Classification Pipeline ===\n")

    # Load data
    train_examples, test_examples = load_data()
    cf_lookup = load_cf_lookup()

    if max_test:
        test_examples = test_examples[:max_test]
        print(f"Running on {max_test} test examples (subset mode)\n")
    else:
        print(f"Running on full test set: {len(test_examples)} examples\n")

    # Build embedding index
    train_embeddings = build_train_index(train_examples)

    # Determine which modes to run
    modes = ["zero_shot", "few_shot", "proposed"] if mode == "all" else [mode]

    all_results = {}

    for current_mode in modes:
        print(f"\n{'─'*55}")
        print(f"Running: {current_mode.upper()}")
        print(f"{'─'*55}")

        true_labels = []
        pred_labels = []
        total       = len(test_examples)

        for i, ex in enumerate(test_examples):
            text       = ex["text"]
            true_label = ex["label"]
            true_labels.append(true_label)

            # Build prompt based on mode
            if current_mode == "zero_shot":
                prompt = build_zero_shot_prompt(text)

            elif current_mode == "few_shot":
                similar = retrieve_similar(
                    text, train_examples, train_embeddings, cf_lookup, k=3
                )
                prompt = build_few_shot_prompt(text, similar, cf_lookup)

            elif current_mode == "proposed":
                similar = retrieve_similar(
                    text, train_examples, train_embeddings, cf_lookup, k=3
                )
                prompt = build_proposed_prompt(text, similar, cf_lookup)

            # Get prediction
            pred = classify_with_gpt(prompt)
            pred_labels.append(pred)

            # Progress
            if (i + 1) % 100 == 0 or (i + 1) == total:
                correct = sum(p == t for p, t in zip(pred_labels, true_labels))
                print(f"  Progress: {i+1}/{total} "
                      f"| Running acc: {correct/(i+1):.2%}")

            # Small delay to avoid rate limiting
            time.sleep(0.05)

        # Compute and print metrics
        metrics = compute_metrics(true_labels, pred_labels)
        print_results(current_mode, metrics)
        all_results[current_mode] = metrics

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n{'='*75}")
        print(f"  COMPARISON — GPT-4.1-mini on HateXplain Test Set")
        print(f"{'='*75}")
        header = f"  {'Metric':<24}"
        for m in modes:
            header += f"  {m.replace('_', ' ').title():>14}"
        print(header)
        print(f"{'─'*75}")

        metrics_to_show = [
            ("Accuracy",          "accuracy"),
            ("Macro F1",          "macro_f1"),
            ("Normal F1",         "normal_f1"),
            ("Offensive F1",      "offensive_f1"),
            ("Hate Speech F1",    "hatespeech_f1"),
            ("Harmful-subset F1", "harmful_subset_f1"),
            ("Binary F1",         "binary_f1"),
        ]

        for display, key in metrics_to_show:
            vals = [all_results[m][key] for m in modes]
            best = max(vals)
            row  = f"  {display:<24}"
            for v in vals:
                marker = " ★" if abs(v - best) < 1e-6 else "  "
                row   += f"  {v:.4f}{marker}"
            if key == "harmful_subset_f1":
                row += "  ← KEY"
            print(row)

        print(f"{'─'*75}")
        print(f"  ★ = best value for this metric")
        print(f"{'='*75}\n")

    # Save results
    if save_results:
        results_dir = Path(config["paths"]["results"])
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "gpt_pipeline_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {out_path}")

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT-4.1-mini agentic classification pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["zero_shot", "few_shot", "proposed", "all"],
        help="Which pipeline mode to run"
    )
    parser.add_argument(
        "--max_test",
        type=int,
        default=None,
        help="Limit test examples (e.g. 50 for quick test)"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save results to JSON"
    )
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        max_test=args.max_test,
        save_results=not args.no_save,
    )