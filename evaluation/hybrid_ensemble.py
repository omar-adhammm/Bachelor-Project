import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
from openai import OpenAI

from configs.config_loader import load_config
from training.data_loader import load_split
from models.proposed_model import ProposedModel

config = load_config()
device = "cpu"

LABEL_MAP     = {0: "normal", 1: "offensive", 2: "hatespeech"}
LABEL_MAP_REV = {"normal": 0, "offensive": 1, "hatespeech": 2}


# ── Dataset ───────────────────────────────────────────────────────────────────
class SimpleDataset(Dataset):
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


# ── Get predictions and probabilities ────────────────────────────────────────
def get_predictions_and_probs(model, examples, tokenizer, device="cpu"):
    dataset = SimpleDataset(examples, tokenizer, max_length=128)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Model inference", leave=False):
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device))
            probs = torch.softmax(output["logits"], dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


# ── GPT classification ────────────────────────────────────────────────────────
def gpt_classify(text, client, model="gpt-4o"):
    system_prompt = """You are an expert hate speech classifier for social media posts from Twitter and Gab.
You will classify posts into exactly one of three categories based on the HateXplain annotation guidelines.

DEFINITIONS:
- normal: Content that is not harmful. May discuss sensitive topics, contain strong opinions, 
  or even mention slurs in a non-harmful context (e.g. quoting someone, academic discussion).
- offensive: Contains language that is rude, vulgar, or hurtful to individuals or groups, 
  but does not specifically target a group with discriminatory intent based on protected characteristics.
- hatespeech: Specifically targets a group based on race, ethnicity, religion, gender, 
  sexual orientation, disability, or national origin with discriminatory, dehumanising, 
  or hateful intent.

KEY DISTINCTION: offensive content may use strong language but lacks targeted group discrimination. 
Hatespeech specifically attacks a protected group. When uncertain between offensive and hatespeech, 
ask: is a specific protected group being attacked or dehumanised?

FEW-SHOT EXAMPLES:

Post: "have a nice day at work white man"
Reasoning: This addresses someone by race in a neutral greeting context. No harmful intent.
Label: normal

Post: "these politicians are all corrupt lying bastards who should be thrown out"
Reasoning: Offensive language directed at politicians, but politicians are not a protected group. No group discrimination.
Label: offensive

Post: "women are too emotional to be good leaders and should stay home"
Reasoning: Directly targets women as a protected group with a discriminatory stereotype about their capabilities.
Label: hatespeech

Post: "i hate mondays so much i could scream"
Reasoning: Expression of frustration about a day of the week. No harmful content.
Label: normal

Post: "shut the hell up you absolute moron nobody asked for your opinion"
Reasoning: Rude and aggressive toward an individual but not targeting a protected group.
Label: offensive

Post: "those people have always been criminals by nature its just how they are"
Reasoning: Implies an ethnic or racial group has criminal nature by birth. Dehumanising stereotype targeting a group.
Label: hatespeech"""

    user_prompt = f"""Classify this social media post.

Post: "{text}"

First write one sentence of reasoning, then on a new line write ONLY the label word.
Format:
Reasoning: [one sentence]
Label: [normal/offensive/hatespeech]"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0)

        content = response.choices[0].message.content.strip().lower()

        # Extract label from response
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("label:"):
                label_str = line.replace("label:", "").strip()
                if label_str in LABEL_MAP_REV:
                    return LABEL_MAP_REV[label_str]

        # Fallback: check if any label word appears in the response
        for label_str in ["hatespeech", "offensive", "normal"]:
            if label_str in content:
                return LABEL_MAP_REV[label_str]

        return None

    except Exception as e:
        print(f"GPT error: {e}")
        return None


# ── Compute metrics ───────────────────────────────────────────────────────────
def compute_metrics(preds, labels):
    per_class = f1_score(labels, preds, average=None,
                         labels=[0,1,2], zero_division=0)
    harmful_mask = labels != 0
    harmful_f1 = f1_score(
        labels[harmful_mask], preds[harmful_mask],
        average="macro", zero_division=0
    ) if harmful_mask.sum() > 0 else 0.0
    return {
        "accuracy":          float(accuracy_score(labels, preds)),
        "macro_f1":          float(f1_score(labels, preds,
                                             average="macro", zero_division=0)),
        "normal_f1":         float(per_class[0]),
        "offensive_f1":      float(per_class[1]),
        "hatespeech_f1":     float(per_class[2]),
        "harmful_subset_f1": float(harmful_f1),
        "binary_f1":         float(f1_score(
            (labels != 0).astype(int),
            (preds  != 0).astype(int),
            average="binary", zero_division=0)),
    }


def print_metrics(metrics, name):
    print(f"\n  {'='*50}")
    print(f"  {name}")
    print(f"  {'='*50}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Normal F1:         {metrics['normal_f1']:.4f}")
    print(f"  Offensive F1:      {metrics['offensive_f1']:.4f}")
    print(f"  Hate Speech F1:    {metrics['hatespeech_f1']:.4f}")
    print(f"  Harmful-subset F1: {metrics['harmful_subset_f1']:.4f}")
    print(f"  Binary F1:         {metrics['binary_f1']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key",   type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Confidence threshold below which GPT is called")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use")
    args = parser.parse_args()

    print(f"Confidence threshold: {args.threshold}")
    print(f"GPT model: {args.gpt_model}")

    checkpoint_dir = Path(config["paths"]["checkpoints"])
    results_dir    = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data and model
    test_examples = load_split("test")
    tokenizer     = AutoTokenizer.from_pretrained(
        config["models"]["hatebert"]["name"])
    model         = load_model(
        ProposedModel,
        checkpoint_dir / "proposed_epoch_31_loss_0.6994.pt",
        device)

    # Get model predictions and probabilities
    print("\nRunning model inference...")
    preds, labels, probs = get_predictions_and_probs(
        model, test_examples, tokenizer, device)

    # Baseline metrics (proposed model alone)
    baseline_metrics = compute_metrics(preds, labels)
    print_metrics(baseline_metrics, "Proposed Model (no ensemble)")

    # Find low-confidence examples
    max_probs = probs.max(axis=1)
    low_conf_mask = max_probs < args.threshold
    low_conf_indices = np.where(low_conf_mask)[0]

    print(f"\nLow-confidence examples (conf < {args.threshold}): "
          f"{len(low_conf_indices)} out of {len(test_examples)} "
          f"({100*len(low_conf_indices)/len(test_examples):.1f}%)")

    # Call GPT on low-confidence examples
    client = OpenAI(api_key=args.api_key)
    cache_path = results_dir / "hybrid_cache.json"

    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached GPT results")

    hybrid_preds = preds.copy()
    gpt_calls    = 0
    gpt_changed  = 0

    for idx in tqdm(low_conf_indices, desc="GPT classification"):
        ex   = test_examples[idx]
        text = ex["text"]

        if text in cache:
            gpt_label = cache[text]
        else:
            gpt_label = gpt_classify(text, client, args.gpt_model)
            if gpt_label is not None:
                cache[text] = gpt_label
            gpt_calls += 1

            # Save cache every 50 calls
            if gpt_calls % 50 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)

        if gpt_label is not None:
            if gpt_label != hybrid_preds[idx]:
                gpt_changed += 1
            hybrid_preds[idx] = gpt_label

    # Final cache save
    with open(cache_path, "w") as f:
        json.dump(cache, f)

    print(f"\nGPT called on {gpt_calls} new examples")
    print(f"GPT changed {gpt_changed} predictions "
          f"({100*gpt_changed/max(len(low_conf_indices),1):.1f}% of low-conf)")

    # Hybrid metrics
    hybrid_metrics = compute_metrics(hybrid_preds, labels)
    print_metrics(hybrid_metrics, f"Hybrid Ensemble (threshold={args.threshold})")

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Proposed':>12} {'Hybrid':>12} {'Delta':>10}")
    print(f"{'─'*60}")
    for k in ["accuracy", "macro_f1", "normal_f1", "offensive_f1",
              "hatespeech_f1", "harmful_subset_f1", "binary_f1"]:
        base = baseline_metrics[k]
        hyb  = hybrid_metrics[k]
        delta = hyb - base
        marker = " +" if delta > 0 else ""
        print(f"  {k:<25} {base:>12.4f} {hyb:>12.4f} "
              f"{delta:>+10.4f}{marker}")
    print(f"{'='*60}")

    # Try multiple thresholds
    print(f"\n{'='*60}")
    print(f"  THRESHOLD SWEEP (using cached GPT results)")
    print(f"{'='*60}")
    print(f"  {'Threshold':<12} {'Low-conf N':>12} {'Macro F1':>10} "
          f"{'Off F1':>10} {'Hate F1':>10}")
    print(f"{'─'*60}")

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask    = probs.max(axis=1) < thresh
        indices = np.where(mask)[0]
        t_preds = preds.copy()
        for idx in indices:
            text = test_examples[idx]["text"]
            if text in cache:
                t_preds[idx] = cache[text]
        m = compute_metrics(t_preds, labels)
        print(f"  {thresh:<12.2f} {len(indices):>12} "
              f"{m['macro_f1']:>10.4f} "
              f"{m['offensive_f1']:>10.4f} "
              f"{m['hatespeech_f1']:>10.4f}")

    print(f"{'='*60}")

    # Save results
    save_path = results_dir / "hybrid_ensemble_results.json"
    with open(save_path, "w") as f:
        json.dump({
            "threshold":        args.threshold,
            "gpt_model":        args.gpt_model,
            "low_conf_count":   len(low_conf_indices),
            "gpt_calls":        gpt_calls,
            "gpt_changed":      gpt_changed,
            "baseline_metrics": baseline_metrics,
            "hybrid_metrics":   hybrid_metrics,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")