import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
import json
import shutil
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from configs.config_loader import load_config
from training.trainer import ModelTrainer
from training.data_loader import load_split, HateXplainDataset

config  = load_config()
SEEDS   = [42, 123, 456]
MODELS  = ["ablation", "proposed"]
DRIVE_PATH = "/content/drive/MyDrive/thesis_v3/multiseed_ablation_proposed.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_test_metrics(model, model_name, device):
    """Evaluate model on test set and return metrics."""
    if model_name == "bert":
        from models.bert_baseline import get_tokenizer
    else:
        from models.hatebert_baseline import get_tokenizer
    tokenizer    = get_tokenizer()
    test_examples = load_split("test")
    test_dataset  = HateXplainDataset(
        test_examples, tokenizer,
        max_length=config["models"]["hatebert"]["max_length"])
    loader = DataLoader(test_dataset, batch_size=32,
                        shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device))
            preds = torch.argmax(
                torch.softmax(output["logits"], dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    harmful_mask = all_labels != 0
    harmful_f1 = f1_score(
        all_labels[harmful_mask], all_preds[harmful_mask],
        average="macro", zero_division=0
    ) if harmful_mask.sum() > 0 else 0.0

    return {
        "accuracy":          float(accuracy_score(all_labels, all_preds)),
        "macro_f1":          float(f1_score(all_labels, all_preds,
                                             average="macro", zero_division=0)),
        "normal_f1":         float(per_class[0]),
        "offensive_f1":      float(per_class[1]),
        "hatespeech_f1":     float(per_class[2]),
        "harmful_subset_f1": float(harmful_f1),
        "binary_f1":         float(f1_score(
            (all_labels != 0).astype(int),
            (all_preds  != 0).astype(int),
            average="binary", zero_division=0)),
    }


def save_progress(all_results):
    results_path = Path(config["paths"]["results"]) / "multiseed_ablation_proposed.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    try:
        shutil.copy(str(results_path), DRIVE_PATH)
        print(f"  Saved to Drive")
    except Exception as e:
        print(f"  Could not save to Drive: {e}")


def load_progress():
    drive_path = Path(DRIVE_PATH)
    if drive_path.exists():
        print("Found partial results, resuming...")
        with open(drive_path) as f:
            return json.load(f)
    return {"per_seed": {}, "aggregated": {}}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Models: {MODELS}")
    print(f"Seeds:  {SEEDS}")

    all_results = load_progress()

    for model_name in MODELS:
        if model_name not in all_results["per_seed"]:
            all_results["per_seed"][model_name] = []

        completed_seeds = [r["seed"] for r in
                           all_results["per_seed"][model_name]]

        for seed in SEEDS:
            if seed in completed_seeds:
                print(f"\nSkipping {model_name} seed={seed} (already done)")
                continue

            print(f"\n{'='*60}")
            print(f"  {model_name.upper()} | seed={seed}")
            print(f"{'='*60}")

            set_seed(seed)

            # Use your real trainer exactly as it was used originally
            trainer = ModelTrainer(device=device)
            trainer.setup_models(model_name=model_name)
            trainer.setup_data(model_name=model_name)
            trainer.setup_optimizers()

            # Set epochs from config
            epochs = {
                "ablation": 50,
                "proposed": 50,
            }[model_name]

            trainer.train(num_epochs=epochs, model_name=model_name)

            # Get test metrics using the best checkpoint
            best_ckpt = None
            best_f1   = -1.0
            ckpt_dir  = Path(config["paths"]["checkpoints"])
            for ckpt in ckpt_dir.glob(f"{model_name}_epoch_*.pt"):
                checkpoint = torch.load(ckpt, map_location=device,
                                        weights_only=False)
                # Use val macro f1 stored in best_metrics
                pass

            # Load best state from trainer directly
            model    = trainer.models[model_name]
            metrics  = get_test_metrics(model, model_name, device)
            best_val = trainer.best_metrics[model_name].get(
                "best_val_macro_f1", 0.0)
            metrics["best_val_macro_f1"] = float(best_val)

            print(f"\n  Test macro F1: {metrics['macro_f1']:.4f}")
            print(f"  Best val macro F1: {best_val:.4f}")

            all_results["per_seed"][model_name].append({
                "seed":    seed,
                "metrics": metrics,
            })

            save_progress(all_results)

            # Free memory
            del trainer
            torch.cuda.empty_cache()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    metric_keys = ["accuracy", "macro_f1", "normal_f1", "offensive_f1",
                   "hatespeech_f1", "harmful_subset_f1", "binary_f1"]

    for model_name in MODELS:
        runs = all_results["per_seed"][model_name]
        agg  = {}
        for k in metric_keys:
            vals   = [r["metrics"][k] for r in runs]
            agg[k] = {"mean": float(np.mean(vals)),
                      "std":  float(np.std(vals))}
        all_results["aggregated"][model_name] = agg

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MULTI-SEED RESULTS — ABLATION AND PROPOSED")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'Ablation':>20} {'Proposed':>20}")
    print(f"{'─'*70}")

    for k in metric_keys:
        row = f"  {k:<25}"
        for mn in MODELS:
            if mn in all_results["aggregated"]:
                m   = all_results["aggregated"][mn][k]["mean"]
                s   = all_results["aggregated"][mn][k]["std"]
                row += f"  {m:.4f}+/-{s:.4f}"
        print(row)

    print(f"{'='*70}")

    save_progress(all_results)
    print("\nDone.")
    