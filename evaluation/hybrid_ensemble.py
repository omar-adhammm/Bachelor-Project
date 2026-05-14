# evaluation/hybrid_ensemble.py
"""
Hybrid ensemble: BERTweet for confident predictions + GPT-4.1-mini for uncertain cases.
Uses confidence threshold to decide which model classifies each example.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from configs.config_loader import load_config
from models.proposed_model import ProposedModel, get_tokenizer
from transformers import AutoTokenizer

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except ImportError:
    raise ImportError("Run: pip install openai")

try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except ImportError:
    raise ImportError("Run: pip install sentence-transformers")

config     = load_config()
LABEL_NAMES = ["normal", "offensive", "hatespeech"]
MODEL_NAME  = "gpt-4.1-mini"
CONFIDENCE_THRESHOLD = 0.60  # BERTweet predictions below this go to GPT


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_data():
    data_dir = Path(config["paths"]["data"])
    with open(data_dir / "test.json", encoding="utf-8") as f:
        return json.load(f)

def load_train_data():
    data_dir = Path(config["paths"]["data"])
    with open(data_dir / "train.json", encoding="utf-8") as f:
        return json.load(f)

def load_cf_lookup():
    cf_path = Path(config["paths"]["cf_pairs"]) / "train_cf_pairs.jsonl"
    lookup  = {}
    if not cf_path.exists():
        return lookup
    with open(cf_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            lookup[pair["original"]["id"]] = pair["counterfactual"]["text"]
    print(f"Loaded {len(lookup)} CF pairs")
    return lookup

def get_rationale_tokens(example):
    text   = example.get("text", "")
    tokens = text.split()
    mask   = example.get("rationale_mask", [])
    if not mask or not any(mask):
        return []
    return [tokens[i] for i, val in enumerate(mask)
            if val == 1 and i < len(tokens)]


# ── BERTweet inference ────────────────────────────────────────────────────────

def load_bertweet_model(checkpoint_path: str, device: str):
    """Load BERTweet proposed model from checkpoint."""
    print(f"Loading BERTweet from: {checkpoint_path}")
    model = ProposedModel(num_labels=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("BERTweet loaded successfully")
    return model

def bertweet_predict_batch(
    model,
    tokenizer,
    texts:  list,
    device: str,
    batch_size: int = 32,
) -> tuple:
    """
    Run BERTweet inference on a list of texts.
    Returns (predictions, confidences) — both lists of length len(texts).
    """
    all_preds  = []
    all_confs  = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoding    = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config["models"]["hatebert"]["max_length"],
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask)
        
        probs = torch.softmax(output["logits"], dim=1)
        preds = torch.argmax(probs, dim=1)
        confs = probs.max(dim=1).values

        all_preds.extend(preds.cpu().numpy().tolist())
        all_confs.extend(confs.cpu().numpy().tolist())

    return all_preds, all_confs


# ── Embedding index for GPT few-shot retrieval ────────────────────────────────

def build_train_index(train_examples):
    print("Building embedding index...")
    texts      = [ex["text"] for ex in train_examples]
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

def retrieve_one_per_class(
    query_text:       str,
    train_examples:   list,
    train_embeddings: np.ndarray,
) -> list:
    """Retrieve one example per class by cosine similarity."""
    query_emb = embedder.encode([query_text])
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    scores    = (train_embeddings @ query_emb.T).squeeze()

    selected = []
    for target_label in ["normal", "offensive", "hatespeech"]:
        class_indices = [i for i, ex in enumerate(train_examples)
                        if ex.get("label") == target_label]
        if not class_indices:
            continue
        best_idx = max(class_indices, key=lambda i: scores[i])
        selected.append(train_examples[best_idx])
    return selected


# ── GPT prompt for uncertain cases ───────────────────────────────────────────

def build_uncertain_prompt(
    text:             str,
    bertweet_probs:   list,
    similar_examples: list,
    cf_lookup:        dict,
) -> str:
    """
    Targeted prompt for uncertain cases.
    Tells GPT what BERTweet thinks and asks it to resolve the ambiguity.
    Includes CF pairs and rationales to help with the decision.
    """
    # Format BERTweet probability distribution
    prob_str = (
        f"normal: {bertweet_probs[0]:.1%}, "
        f"offensive: {bertweet_probs[1]:.1%}, "
        f"hatespeech: {bertweet_probs[2]:.1%}"
    )

    # Format few-shot examples with CF and rationale
    examples_str = ""
    for i, ex in enumerate(similar_examples, 1):
        ex_id      = ex.get("id", "")
        cf_text    = cf_lookup.get(ex_id, None)
        rat_tokens = get_rationale_tokens(ex)
        label      = ex.get("label", "normal")

        examples_str += f"Example {i}:\n"
        examples_str += f"Text: {ex['text']}\n"
        if rat_tokens:
            examples_str += (
                f"Key harmful phrases (identified by human annotators): "
                f"{rat_tokens}\n"
            )
        if cf_text and label != "normal":
            examples_str += (
                f"Neutral rewrite: {cf_text}\n"
                f"(The neutral rewrite removes only the harmful element — "
                f"the difference reveals exactly what makes the original harmful)\n"
            )
        examples_str += f"Label: {label}\n\n"

    return (
        "You are an expert hate speech classifier resolving an ambiguous case.\n\n"
        "Categories:\n"
        "- normal: harmless text, no offensive or hateful content\n"
        "- offensive: rude or disrespectful text NOT targeting a protected group\n"
        "- hatespeech: attacks or demeans a group based on race, religion, "
        "gender, ethnicity, or other protected characteristics\n\n"
        f"A classifier is uncertain about this text (probabilities: {prob_str}):\n"
        f"Text: {text}\n\n"
        "Here are similar examples with annotations to help you decide:\n\n"
        f"{examples_str}"
        "Resolve the ambiguity:\n"
        "Step 1: Does this text target a protected group (race, religion, "
        "gender, ethnicity)? If yes → hatespeech.\n"
        "Step 2: Is it merely rude without group targeting? If yes → offensive.\n"
        "Step 3: Is it harmless? If yes → normal.\n\n"
        "Respond with exactly one word: normal, offensive, or hatespeech."
    )


# ── GPT inference ─────────────────────────────────────────────────────────────

def classify_with_gpt(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower()
            raw_clean = raw.replace("-", "").replace("_", "").strip()

            if "hatespeech" in raw_clean or "hate speech" in raw_clean:
                return "hatespeech"
            elif "offensive" in raw_clean:
                return "offensive"
            elif "normal" in raw_clean:
                return "normal"
            else:
                if "hate" in raw_clean:
                    return "hatespeech"
                elif "offens" in raw_clean:
                    return "offensive"
                else:
                    return "normal"

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return "normal"


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(true_labels: list, pred_labels: list) -> dict:
    label2id = {"normal": 0, "offensive": 1, "hatespeech": 2}
    y_true   = [label2id[l] for l in true_labels]
    y_pred   = [label2id[l] for l in pred_labels]

    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class   = f1_score(y_true, y_pred, average=None,       zero_division=0)
    accuracy    = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    harmful_mask = [i for i, l in enumerate(y_true) if l > 0]
    harmful_f1   = f1_score(
        [y_true[i] for i in harmful_mask],
        [y_pred[i] for i in harmful_mask],
        average="macro", zero_division=0,
    )

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
    print(f"\n{'='*57}")
    print(f"  {name.upper()} — Evaluation Results")
    print(f"{'='*57}")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}  "
          f"({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:         {metrics['weighted_f1']:.4f}")
    print(f"{'─'*57}")
    print(f"  Per-class F1:")
    print(f"    Normal:            {metrics['normal_f1']:.4f}")
    print(f"    Offensive:         {metrics['offensive_f1']:.4f}")
    print(f"    Hate Speech:       {metrics['hatespeech_f1']:.4f}")
    print(f"{'─'*57}")
    print(f"  Harmful-subset F1:   {metrics['harmful_subset_f1']:.4f}  ← KEY")
    print(f"  Binary F1:           {metrics['binary_f1']:.4f}")
    print(f"{'='*57}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_hybrid_ensemble(
    checkpoint_path: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    device: str = "cpu",
):
    print("=== Hybrid Ensemble: BERTweet + GPT-4.1-mini ===\n")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Examples below threshold go to GPT\n")

    # Load everything
    test_examples  = load_test_data()
    train_examples = load_train_data()
    cf_lookup      = load_cf_lookup()
    tokenizer      = get_tokenizer()
    model          = load_bertweet_model(checkpoint_path, device)
    train_embeddings = build_train_index(train_examples)

    texts      = [ex["text"]  for ex in test_examples]
    true_labels = [ex["label"] for ex in test_examples]

    # Step 1 — BERTweet predictions on full test set
    print("Step 1: Running BERTweet on full test set...")
    bertweet_preds, bertweet_confs = bertweet_predict_batch(
        model, tokenizer, texts, device
    )
    bertweet_labels = [LABEL_NAMES[p] for p in bertweet_preds]

    # Compute BERTweet-only metrics
    bertweet_metrics = compute_metrics(true_labels, bertweet_labels)
    print_results("BERTweet Only", bertweet_metrics)

    # Step 2 — identify uncertain examples
    uncertain_indices = [
        i for i, conf in enumerate(bertweet_confs)
        if conf < confidence_threshold
    ]
    confident_indices = [
        i for i, conf in enumerate(bertweet_confs)
        if conf >= confidence_threshold
    ]

    print(f"Step 2: Confidence analysis")
    print(f"  Confident (BERTweet keeps):  {len(confident_indices)} "
          f"({len(confident_indices)/len(texts):.1%})")
    print(f"  Uncertain (GPT resolves):    {len(uncertain_indices)} "
          f"({len(uncertain_indices)/len(texts):.1%})\n")

    # Step 3 — run GPT on uncertain examples only
    print("Step 3: Running GPT-4.1-mini on uncertain examples...")
    hybrid_labels = bertweet_labels.copy()

    # Get full probability distributions for uncertain examples
    print("  Getting probability distributions for uncertain examples...")
    uncertain_texts = [texts[i] for i in uncertain_indices]

    # Run BERTweet again on uncertain examples to get full prob distributions
    all_probs = []
    for i in range(0, len(uncertain_texts), 32):
        batch = uncertain_texts[i:i+32]
        encoding = tokenizer(
            batch, padding=True, truncation=True,
            max_length=config["models"]["hatebert"]["max_length"],
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(
                encoding["input_ids"].to(device),
                encoding["attention_mask"].to(device),
            )
        probs = torch.softmax(output["logits"], dim=1)
        all_probs.extend(probs.cpu().numpy().tolist())

    # Call GPT for each uncertain example
    gpt_used    = 0
    gpt_changed = 0

    for idx, (orig_idx, probs) in enumerate(
        zip(uncertain_indices, all_probs)
    ):
        text = texts[orig_idx]

        # Retrieve similar examples for context
        similar = retrieve_one_per_class(text, train_examples, train_embeddings)

        # Build targeted prompt
        prompt = build_uncertain_prompt(text, probs, similar, cf_lookup)

        # Get GPT prediction
        gpt_pred = classify_with_gpt(prompt)
        gpt_used += 1

        if gpt_pred != bertweet_labels[orig_idx]:
            gpt_changed += 1

        hybrid_labels[orig_idx] = gpt_pred

        if (idx + 1) % 100 == 0:
            correct = sum(
                hybrid_labels[i] == true_labels[i]
                for i in range(len(true_labels))
            )
            print(f"  GPT progress: {idx+1}/{len(uncertain_indices)} "
                  f"| Running hybrid acc: {correct/len(true_labels):.2%}")

        time.sleep(0.05)

    print(f"\n  GPT calls made: {gpt_used}")
    print(f"  GPT changed BERTweet prediction: {gpt_changed} "
          f"({gpt_changed/max(gpt_used,1):.1%} of uncertain cases)\n")

    # Step 4 — compute hybrid metrics
    hybrid_metrics = compute_metrics(true_labels, hybrid_labels)
    print_results("Hybrid Ensemble (BERTweet + GPT)", hybrid_metrics)

    # Comparison
    print(f"\n{'='*65}")
    print(f"  COMPARISON")
    print(f"{'='*65}")
    metrics_to_show = [
        ("Accuracy",          "accuracy"),
        ("Macro F1",          "macro_f1"),
        ("Normal F1",         "normal_f1"),
        ("Offensive F1",      "offensive_f1"),
        ("Hate Speech F1",    "hatespeech_f1"),
        ("Harmful-subset F1", "harmful_subset_f1"),
        ("Binary F1",         "binary_f1"),
    ]
    print(f"  {'Metric':<24} {'BERTweet':>12} {'Hybrid':>12} {'Delta':>10}")
    print(f"{'─'*65}")
    for display, key in metrics_to_show:
        b = bertweet_metrics[key]
        h = hybrid_metrics[key]
        delta = h - b
        marker = " ★" if delta > 0 else "  "
        note = "← KEY" if key == "harmful_subset_f1" else ""
        print(f"  {display:<24} {b:>12.4f} {h:>12.4f} "
              f"{delta:>+10.4f}{marker} {note}")
    print(f"{'='*65}\n")

    # Save results
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "hybrid_ensemble_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "bertweet_only": bertweet_metrics,
            "hybrid":        hybrid_metrics,
            "config": {
                "confidence_threshold": confidence_threshold,
                "uncertain_count":      len(uncertain_indices),
                "confident_count":      len(confident_indices),
                "gpt_changed":          gpt_changed,
            }
        }, f, indent=2)
    print(f"Results saved to: {out_path}")

    return bertweet_metrics, hybrid_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid BERTweet + GPT ensemble for hate speech detection"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to BERTweet proposed model checkpoint"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Confidence threshold — below this GPT is used (default: 0.60)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda"
    )
    args = parser.parse_args()

    run_hybrid_ensemble(
        checkpoint_path=args.checkpoint,
        confidence_threshold=args.threshold,
        device=args.device,
    )
