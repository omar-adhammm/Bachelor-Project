# evaluation/metrics.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, accuracy_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_loader import load_config
from training.data_loader import load_split, HateXplainDataset

config = load_config()
LABEL_NAMES = ["normal", "offensive", "hatespeech"]


# ── Load model from checkpoint ────────────────────────────────────────────────

def load_model_from_checkpoint(model_class, checkpoint_path: str, device: str = "cpu"):
    """Load a trained model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_class(num_labels=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Saved at epoch: {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f}")
    return model


# ── Run inference on a split ──────────────────────────────────────────────────

def get_predictions(model, split: str, device: str = "cpu", model_name: str = "baseline") -> tuple:
    """
    Run model on a dataset split and return predictions + true labels.
    Returns: (all_preds, all_labels, all_probs)
    """
    from transformers import AutoTokenizer

    if model_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["models"]["hatebert"]["name"])

    examples = load_split(split)
    dataset  = HateXplainDataset(
        examples, tokenizer,
        max_length=config["models"]["hatebert"]["max_length"]
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Running inference on {split}"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            output = model(input_ids, attention_mask)
            probs  = torch.softmax(output["logits"], dim=1)
            preds  = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute all metrics for a set of predictions.
    Returns a dict with all metrics.
    """
    accuracy    = accuracy_score(labels, preds)
    macro_f1    = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    harmful_mask   = labels != 0
    harmful_preds  = preds[harmful_mask]
    harmful_labels = labels[harmful_mask]

    harmful_f1 = f1_score(
        harmful_labels, harmful_preds,
        average="macro", zero_division=0
    ) if len(harmful_labels) > 0 else 0.0

    binary_labels = (labels != 0).astype(int)
    binary_preds  = (preds  != 0).astype(int)
    binary_f1     = f1_score(binary_labels, binary_preds, average="binary", zero_division=0)

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    return {
        "accuracy":          float(accuracy),
        "macro_f1":          float(macro_f1),
        "weighted_f1":       float(weighted_f1),
        "normal_f1":         float(per_class_f1[0]),
        "offensive_f1":      float(per_class_f1[1]),
        "hatespeech_f1":     float(per_class_f1[2]),
        "harmful_subset_f1": float(harmful_f1),
        "binary_f1":         float(binary_f1),
        "confusion_matrix":  cm.tolist(),
    }


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Print a clean metrics report."""
    print(f"\n{'='*55}")
    print(f"  {model_name.upper()} — Evaluation Results")
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
    print(f"{'─'*55}")
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(f"               Normal  Offens  Hate")
    cm = metrics["confusion_matrix"]
    for i, row_name in enumerate(["Normal ", "Offens ", "Hate   "]):
        print(f"    {row_name}  {cm[i][0]:6d}  {cm[i][1]:6d}  {cm[i][2]:6d}")
    print(f"{'='*55}\n")


# ── Compare all models ────────────────────────────────────────────────────────

def compare_models(results: dict):
    """Print a side-by-side comparison table of all models."""
    print(f"\n{'='*85}")
    print(f"  CROSS-MODEL COMPARISON — TEST SET")
    print(f"{'='*85}")
    print(f"  {'Metric':<25} {'BERT':>10} {'Baseline':>10} {'Ablation':>10} {'Proposed':>10}")
    print(f"{'─'*85}")

    metrics_to_show = [
        ("Accuracy",          "accuracy"),
        ("Macro F1",          "macro_f1"),
        ("Normal F1",         "normal_f1"),
        ("Offensive F1",      "offensive_f1"),
        ("Hate Speech F1",    "hatespeech_f1"),
        ("Harmful-subset F1", "harmful_subset_f1"),
        ("Binary F1",         "binary_f1"),
    ]

    for display_name, key in metrics_to_show:
        vals = {m: results[m][key] for m in results if key in results[m]}
        row  = f"  {display_name:<25}"

        for model_name in ["bert", "baseline", "ablation", "proposed"]:
            if model_name in vals:
                val  = vals[model_name]
                best = max(vals.values())
                marker = " *" if abs(val - best) < 1e-6 and len(vals) > 1 else "  "
                row += f"  {val:.4f}{marker}"
            else:
                row += f"  {'—':>10}"

        if key == "harmful_subset_f1":
            row += "  ← KEY"

        print(row)

    print(f"{'='*85}")
    print(f"  * = best value for this metric")
    print(f"{'='*85}\n")


# ── Main evaluation runner ────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from pathlib import Path

    from models.hatebert_baseline import HateBERTBaseline
    from models.proposed_model    import ProposedModel
    from models.ablation_cf_only  import AblationCFOnlyModel
    from models.bert_baseline     import BERTBaseline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    checkpoint_dir = Path(config["paths"]["checkpoints"])

    def find_best_checkpoint(model_prefix: str) -> str:
        checkpoints = list(checkpoint_dir.glob(f"{model_prefix}_epoch_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for {model_prefix}")
        best = min(checkpoints, key=lambda p: float(p.stem.split("loss_")[1]))
        return str(best)

    model_configs = [
        ("bert",     BERTBaseline,        find_best_checkpoint("bert")),
        ("baseline", HateBERTBaseline,    find_best_checkpoint("baseline")),
        ("ablation", AblationCFOnlyModel, find_best_checkpoint("ablation")),
        ("proposed", ProposedModel,       find_best_checkpoint("proposed")),
    ]

    all_results = {}

    for model_name, model_class, ckpt_path in model_configs:
        print(f"\n{'─'*55}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'─'*55}")

        model = load_model_from_checkpoint(model_class, ckpt_path, device)

        preds, labels, probs = get_predictions(model, "test", device, model_name=model_name)
        metrics = compute_metrics(preds, labels)
        print_metrics(metrics, model_name)

        all_results[model_name] = metrics

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    compare_models(all_results)

    results_path = Path(config["paths"]["results"]) / "final_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_path}")