import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from openai import OpenAI

from configs.config_loader import load_config
from training.data_loader import load_split
from models.bert_baseline import BERTBaseline
from models.hatebert_baseline import HateBERTBaseline
from models.proposed_model import ProposedModel
from models.ablation_cf_only import AblationCFOnlyModel

config = load_config()
device = "cpu"
LABEL_MAP = {0: "normal", 1: "offensive", 2: "hatespeech"}


# ── Dataset ───────────────────────────────────────────────────────────────────
class HateXplainDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex  = self.examples[idx]
        enc = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(ex["label_id"], dtype=torch.long),
            "text":           ex["text"],
        }


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(model_class, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model = model_class(num_labels=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded: {checkpoint_path}")
    return model


# ── Get predictions ───────────────────────────────────────────────────────────
def get_predictions(model, examples, tokenizer, device="cpu"):
    dataset = HateXplainDataset(examples, tokenizer, max_length=128)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Inference", leave=False):
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device))
            preds = torch.argmax(
                torch.softmax(output["logits"], dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())
    return np.array(all_preds), np.array(all_labels)


# ── Classify implicit vs explicit using GPT ───────────────────────────────────
def classify_implicit_explicit(examples, api_key, cache_path="outputs/results/implicit_explicit_cache.json"):
    """
    Use GPT-4o-mini to classify each harmful test example as implicit or explicit.
    Caches results to avoid rerunning if interrupted.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cache if exists
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached classifications")

    client = OpenAI(api_key=api_key)

    # Only classify harmful examples (offensive + hatespeech)
    harmful = [ex for ex in examples if ex["label_id"] in (1, 2)]
    print(f"Classifying {len(harmful)} harmful examples as implicit or explicit...")

    results = {}
    for ex in tqdm(harmful, desc="GPT classification"):
        ex_id = ex["id"]
        if ex_id in cache:
            results[ex_id] = cache[ex_id]
            continue

        prompt = f"""Classify the following text as either IMPLICIT or EXPLICIT harmful content.

EXPLICIT: Contains clear slurs, profanity, or directly hateful/offensive language.
IMPLICIT: Harmful meaning is conveyed through stereotypes, implication, sarcasm, or cultural references WITHOUT explicit slurs or profanity.

Text: "{ex['text']}"

Respond with ONLY one word: IMPLICIT or EXPLICIT"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0)
            classification = response.choices[0].message.content.strip().upper()
            if classification not in ("IMPLICIT", "EXPLICIT"):
                classification = "EXPLICIT"  # default if unclear
            results[ex_id] = classification
            cache[ex_id]   = classification
        except Exception as e:
            print(f"Error on {ex_id}: {e}")
            results[ex_id] = "EXPLICIT"
            cache[ex_id]   = "EXPLICIT"

        # Save cache every 50 examples
        if len(results) % 50 == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f)

    # Final cache save
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    implicit_count = sum(1 for v in results.values() if v == "IMPLICIT")
    explicit_count = sum(1 for v in results.values() if v == "EXPLICIT")
    print(f"  Implicit: {implicit_count} ({100*implicit_count/len(results):.1f}%)")
    print(f"  Explicit: {explicit_count} ({100*explicit_count/len(results):.1f}%)")

    return results


# ── Compute metrics on subset ─────────────────────────────────────────────────
def compute_subset_metrics(preds, labels, indices):
    if len(indices) == 0:
        return {"macro_f1": 0.0, "offensive_f1": 0.0, "hatespeech_f1": 0.0, "n": 0}
    sub_preds  = preds[indices]
    sub_labels = labels[indices]
    per_class  = f1_score(sub_labels, sub_preds, average=None,
                          labels=[0,1,2], zero_division=0)
    return {
        "macro_f1":      float(f1_score(sub_labels, sub_preds,
                                        average="macro", zero_division=0)),
        "offensive_f1":  float(per_class[1]),
        "hatespeech_f1": float(per_class[2]),
        "accuracy":      float(accuracy_score(sub_labels, sub_preds)),
        "n":             len(indices),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True,
                        help="OpenAI API key")
    args = parser.parse_args()

    checkpoint_dir = Path(config["paths"]["checkpoints"])
    results_dir    = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load test examples
    test_examples = load_split("test")
    print(f"Test examples: {len(test_examples)}")

    # Step 1: Classify implicit vs explicit using GPT
    classifications = classify_implicit_explicit(
        test_examples, args.api_key)

    # Build index sets
    # Normal examples are excluded from implicit/explicit analysis
    # since they are not harmful
    implicit_indices = []
    explicit_indices = []
    all_harmful_indices = []

    for i, ex in enumerate(test_examples):
        if ex["label_id"] in (1, 2):
            all_harmful_indices.append(i)
            ex_id = ex["id"]
            if classifications.get(ex_id) == "IMPLICIT":
                implicit_indices.append(i)
            else:
                explicit_indices.append(i)

    implicit_indices     = np.array(implicit_indices)
    explicit_indices     = np.array(explicit_indices)
    all_harmful_indices  = np.array(all_harmful_indices)

    print(f"\nIndex breakdown:")
    print(f"  Total test examples: {len(test_examples)}")
    print(f"  Harmful examples:    {len(all_harmful_indices)}")
    print(f"  Implicit harmful:    {len(implicit_indices)}")
    print(f"  Explicit harmful:    {len(explicit_indices)}")

    # Step 2: Run all four models
    model_configs = [
        ("bert",     BERTBaseline,        "bert-base-uncased",
         checkpoint_dir / "bert_epoch_2_loss_0.8036.pt"),
        ("baseline", HateBERTBaseline,    config["models"]["hatebert"]["name"],
         checkpoint_dir / "baseline_epoch_19_loss_0.7006.pt"),
        ("ablation", AblationCFOnlyModel, config["models"]["hatebert"]["name"],
         checkpoint_dir / "ablation_epoch_27_loss_0.7102.pt"),
        ("proposed", ProposedModel,       config["models"]["hatebert"]["name"],
         checkpoint_dir / "proposed_epoch_31_loss_0.6994.pt"),
    ]

    all_results = {}

    for model_name, model_class, tokenizer_name, ckpt_path in model_configs:
        print(f"\n{'='*55}")
        print(f"  {model_name.upper()}")
        print(f"{'='*55}")

        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model     = load_model(model_class, ckpt_path, device)

        preds, labels = get_predictions(model, test_examples, tokenizer, device)

        # Full test set metrics
        full_metrics = {
            "macro_f1":     float(f1_score(labels, preds,
                                           average="macro", zero_division=0)),
            "offensive_f1": float(f1_score(labels, preds,
                                           average=None, zero_division=0)[1]),
            "hatespeech_f1":float(f1_score(labels, preds,
                                           average=None, zero_division=0)[2]),
            "accuracy":     float(accuracy_score(labels, preds)),
            "n":            len(labels),
        }

        # Implicit subset metrics
        implicit_metrics = compute_subset_metrics(preds, labels, implicit_indices)

        # Explicit subset metrics
        explicit_metrics = compute_subset_metrics(preds, labels, explicit_indices)

        all_results[model_name] = {
            "full":     full_metrics,
            "implicit": implicit_metrics,
            "explicit": explicit_metrics,
        }

        print(f"  Full test    — Macro F1: {full_metrics['macro_f1']:.4f}  "
              f"Off F1: {full_metrics['offensive_f1']:.4f}  "
              f"Hate F1: {full_metrics['hatespeech_f1']:.4f}")
        print(f"  Implicit     — Macro F1: {implicit_metrics['macro_f1']:.4f}  "
              f"Off F1: {implicit_metrics['offensive_f1']:.4f}  "
              f"Hate F1: {implicit_metrics['hatespeech_f1']:.4f}  "
              f"(n={implicit_metrics['n']})")
        print(f"  Explicit     — Macro F1: {explicit_metrics['macro_f1']:.4f}  "
              f"Off F1: {explicit_metrics['offensive_f1']:.4f}  "
              f"Hate F1: {explicit_metrics['hatespeech_f1']:.4f}  "
              f"(n={explicit_metrics['n']})")

        del model

    # ── Print comparison tables ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  IMPLICIT HARMFUL CONTENT — Macro F1 comparison")
    print(f"{'='*70}")
    print(f"  {'Model':<15} {'Full':>10} {'Implicit':>10} {'Explicit':>10} "
          f"{'Impl-Expl':>12}")
    print(f"{'─'*70}")
    for mn in ["bert", "baseline", "ablation", "proposed"]:
        if mn not in all_results:
            continue
        r    = all_results[mn]
        diff = r["implicit"]["macro_f1"] - r["explicit"]["macro_f1"]
        print(f"  {mn:<15} "
              f"{r['full']['macro_f1']:>10.4f} "
              f"{r['implicit']['macro_f1']:>10.4f} "
              f"{r['explicit']['macro_f1']:>10.4f} "
              f"{diff:>12.4f}")
    print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"  IMPLICIT HARMFUL CONTENT — Offensive F1 comparison")
    print(f"{'='*70}")
    print(f"  {'Model':<15} {'Full':>10} {'Implicit':>10} {'Explicit':>10}")
    print(f"{'─'*70}")
    for mn in ["bert", "baseline", "ablation", "proposed"]:
        if mn not in all_results:
            continue
        r = all_results[mn]
        print(f"  {mn:<15} "
              f"{r['full']['offensive_f1']:>10.4f} "
              f"{r['implicit']['offensive_f1']:>10.4f} "
              f"{r['explicit']['offensive_f1']:>10.4f}")
    print(f"{'='*70}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = results_dir / "implicit_explicit_results.json"
    with open(save_path, "w") as f:
        json.dump({
            "classifications": classifications,
            "index_counts": {
                "total":    len(test_examples),
                "harmful":  len(all_harmful_indices),
                "implicit": len(implicit_indices),
                "explicit": len(explicit_indices),
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")