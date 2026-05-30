import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (f1_score, accuracy_score,
                             classification_report, confusion_matrix)
from tqdm import tqdm

from configs.config_loader import load_config
from models.bert_baseline      import BERTBaseline
from models.hatebert_baseline  import HateBERTBaseline
from models.proposed_model     import ProposedModel
from models.ablation_cf_only   import AblationCFOnlyModel

config = load_config()
device = "cpu"


# ── ToxiGen Dataset ───────────────────────────────────────────────────────────
class ToxiGenDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples  = examples
        self.tokenizer = tokenizer
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
            "label":          torch.tensor(ex["label"], dtype=torch.long),
            "text":           ex["text"],
        }


# ── Load ToxiGen human-annotated test set ─────────────────────────────────────
def load_toxigen():
    try:
        from datasets import load_dataset
        print("Loading ToxiGen human-annotated test set...")
        dataset = load_dataset("toxigen/toxigen-data", name="annotated",
                               split="test", trust_remote_code=True)

        examples = []
        for row in dataset:
            # Human annotation: toxicity_human is average rating 1-5
            # >= 3.5 is considered toxic by the original paper
            toxicity = row.get("toxicity_human", 0)
            if toxicity is None:
                continue
            label = 1 if float(toxicity) >= 3.5 else 0
            examples.append({
                "text":  row["text"],
                "label": label,
                "group": row.get("target_group", "unknown"),
            })

        print(f"Loaded {len(examples)} ToxiGen examples")
        toxic   = sum(1 for e in examples if e["label"] == 1)
        benign  = sum(1 for e in examples if e["label"] == 0)
        print(f"  Toxic:  {toxic} ({100*toxic/len(examples):.1f}%)")
        print(f"  Benign: {benign} ({100*benign/len(examples):.1f}%)")
        return examples

    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Trying alternative loading method...")
        return load_toxigen_alternative()


def load_toxigen_alternative():
    """Fallback: load from local file if HF download fails."""
    local_path = Path("data/toxigen/toxigen_test.jsonl")
    if not local_path.exists():
        raise FileNotFoundError(
            "ToxiGen data not found. Run: "
            "pip install datasets && "
            "python -c \"from datasets import load_dataset; "
            "load_dataset('toxigen/toxigen-data', name='annotated')\"")

    examples = []
    with open(local_path) as f:
        for line in f:
            row = json.loads(line)
            toxicity = row.get("toxicity_human", 0)
            label    = 1 if float(toxicity) >= 3.5 else 0
            examples.append({
                "text":  row["text"],
                "label": label,
                "group": row.get("target_group", "unknown"),
            })
    print(f"Loaded {len(examples)} ToxiGen examples from local file")
    return examples


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


# ── Run inference ─────────────────────────────────────────────────────────────
def get_predictions(model, examples, tokenizer, device="cpu"):
    dataset = ToxiGenDataset(examples, tokenizer, max_length=128)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Inference", leave=False):
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device))
            probs = torch.softmax(output["logits"], dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ── Compute binary metrics ────────────────────────────────────────────────────
def compute_binary_metrics(preds_3class, labels_binary):
    """
    Map 3-class HateXplain predictions to binary ToxiGen labels.
    normal (0) -> benign (0)
    offensive (1) or hatespeech (2) -> toxic (1)
    """
    binary_preds = (preds_3class != 0).astype(int)

    accuracy  = accuracy_score(labels_binary, binary_preds)
    f1_toxic  = f1_score(labels_binary, binary_preds,
                         pos_label=1, average="binary", zero_division=0)
    f1_benign = f1_score(labels_binary, binary_preds,
                         pos_label=0, average="binary", zero_division=0)
    macro_f1  = f1_score(labels_binary, binary_preds,
                         average="macro", zero_division=0)

    cm = confusion_matrix(labels_binary, binary_preds)

    return {
        "accuracy":   float(accuracy),
        "toxic_f1":   float(f1_toxic),
        "benign_f1":  float(f1_benign),
        "macro_f1":   float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }


def compute_per_group_metrics(preds_3class, examples):
    """Compute toxic F1 per target community group."""
    from collections import defaultdict
    groups = defaultdict(lambda: {"preds": [], "labels": []})

    binary_preds = (preds_3class != 0).astype(int)

    for i, ex in enumerate(examples):
        group = ex["group"]
        groups[group]["preds"].append(binary_preds[i])
        groups[group]["labels"].append(ex["label"])

    group_results = {}
    for group, data in groups.items():
        if len(data["labels"]) < 10:
            continue
        f1 = f1_score(data["labels"], data["preds"],
                      average="binary", zero_division=0)
        group_results[group] = {
            "toxic_f1": float(f1),
            "n":        len(data["labels"]),
        }
    return group_results


def print_metrics(metrics, model_name):
    print(f"\n{'='*55}")
    print(f"  {model_name.upper()} on ToxiGen")
    print(f"{'='*55}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Toxic F1:    {metrics['toxic_f1']:.4f}  <- KEY METRIC")
    print(f"  Benign F1:   {metrics['benign_f1']:.4f}")
    print(f"{'─'*55}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(f"               Benign   Toxic")
    print(f"    Benign    {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"    Toxic     {cm[1][0]:6d}  {cm[1][1]:6d}")
    print(f"{'='*55}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Install datasets if needed
    try:
        import datasets
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets")

    checkpoint_dir = Path(config["paths"]["checkpoints"])
    results_dir    = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load ToxiGen
    examples = load_toxigen()

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
        print(f"\n{'─'*55}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'─'*55}")

        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model     = load_model(model_class, ckpt_path, device)

        preds, labels, probs = get_predictions(
            model, examples, tokenizer, device)

        metrics       = compute_binary_metrics(preds, labels)
        group_metrics = compute_per_group_metrics(preds, examples)

        print_metrics(metrics, model_name)

        print(f"\n  Per-group Toxic F1:")
        for group, gm in sorted(group_metrics.items(),
                                key=lambda x: x[1]["toxic_f1"],
                                reverse=True):
            print(f"    {group:<20} F1={gm['toxic_f1']:.4f} "
                  f"(n={gm['n']})")

        all_results[model_name] = {
            "metrics":      metrics,
            "group_metrics": group_metrics,
        }

        del model

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TOXIGEN ZERO-SHOT RESULTS")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'BERT':>12} {'Baseline':>12} "
          f"{'Ablation':>12} {'Proposed':>12}")
    print(f"{'─'*70}")

    for metric_key, display in [
        ("accuracy",  "Accuracy"),
        ("macro_f1",  "Macro F1"),
        ("toxic_f1",  "Toxic F1 (KEY)"),
        ("benign_f1", "Benign F1"),
    ]:
        row = f"  {display:<20}"
        for mn in ["bert", "baseline", "ablation", "proposed"]:
            if mn in all_results:
                val  = all_results[mn]["metrics"][metric_key]
                best = max(all_results[r]["metrics"][metric_key]
                           for r in all_results)
                marker = " *" if abs(val - best) < 1e-6 else "  "
                row += f"  {val:.4f}{marker}"
            else:
                row += f"  {'N/A':>8}"
        print(row)

    print(f"{'='*70}")
    print(f"  * = best value")
    print(f"{'='*70}")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = results_dir / "toxigen_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {save_path}")