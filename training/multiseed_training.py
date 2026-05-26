# Run this in Google Colab
# Mount drive first:
# from google.drive import drive
# drive.mount('/content/drive')

import sys
import os
sys.path.append('/content/Bachelor-Project')

import torch
import numpy as np
import json
import random
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_loader import load_config
from training.data_loader import load_split, HateXplainDataset, ContrastiveHateDataset
from training.contrastive_loss import CFContrastiveLoss, BoundaryContrastiveLoss
from models.bert_baseline import BERTBaseline
from models.hatebert_baseline import HateBERTBaseline
from models.proposed_model import ProposedModel
from models.ablation_cf_only import AblationCFOnlyModel

# ── Config ────────────────────────────────────────────────────────────────────
config  = load_config()
device  = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS   = [42, 123, 456]
LR      = 2e-5
PATIENCE = 8
print(f"Device: {device}")
print(f"Seeds: {SEEDS}")


# ── Set seed ──────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(preds, labels):
    preds  = np.array(preds)
    labels = np.array(labels)
    macro_f1  = f1_score(labels, preds, average="macro",  zero_division=0)
    per_class = f1_score(labels, preds, average=None,     zero_division=0)
    harmful_mask = labels != 0
    harmful_f1 = f1_score(
        labels[harmful_mask], preds[harmful_mask],
        average="macro", zero_division=0
    ) if harmful_mask.sum() > 0 else 0.0
    binary_f1 = f1_score(
        (labels != 0).astype(int),
        (preds  != 0).astype(int),
        average="binary", zero_division=0)
    return {
        "accuracy":          float(accuracy_score(labels, preds)),
        "macro_f1":          float(macro_f1),
        "normal_f1":         float(per_class[0]),
        "offensive_f1":      float(per_class[1]),
        "hatespeech_f1":     float(per_class[2]),
        "harmful_subset_f1": float(harmful_f1),
        "binary_f1":         float(binary_f1),
    }


# ── Predictions ───────────────────────────────────────────────────────────────
def get_predictions(model, split, device, model_name="baseline"):
    if model_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["models"]["hatebert"]["name"])
    examples = load_split(split)
    dataset  = HateXplainDataset(
        examples, tokenizer,
        max_length=config["models"]["hatebert"]["max_length"])
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
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
    return np.array(all_preds), np.array(all_labels)


def evaluate_val_macro_f1(model, device, model_name="baseline"):
    preds, labels = get_predictions(model, "validation", device, model_name)
    return f1_score(labels, preds, average="macro", zero_division=0)


# ── Build loaders ─────────────────────────────────────────────────────────────
def build_loaders(model_name):
    if model_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["models"]["hatebert"]["name"])

    train_examples = load_split("train")
    train_dataset  = HateXplainDataset(
        train_examples, tokenizer,
        max_length=config["models"]["hatebert"]["max_length"])
    train_loader   = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0)

    cf_loader = None
    if model_name in ("ablation", "proposed"):
        cf_pairs_path = Path(config["paths"]["cf_pairs"]) / "train_cf_pairs.jsonl"
        cf_pairs      = [json.loads(l) for l in open(cf_pairs_path)]

        # Build rationale lookup same way your trainer does
        train_examples_lookup = load_split("train")
        rationale_lookup = {}
        for ex in train_examples_lookup:
            if "rationale" in ex and ex["rationale"]:
                rationale_lookup[ex.get("id", "")] = ex["rationale"]

        cf_dataset = ContrastiveHateDataset(
            cf_pairs, tokenizer,
            max_length=config["models"]["hatebert"]["max_length"],
            rationale_lookup=rationale_lookup)
        cf_loader  = DataLoader(
            cf_dataset, batch_size=32, shuffle=True, num_workers=0)

    return train_loader, cf_loader


# ── Class weights ─────────────────────────────────────────────────────────────
def get_class_weights(device):
    LABEL_MAP = {"normal": 0, "offensive": 1, "hatespeech": 2}
    examples = load_split("train")
    labels   = []
    for e in examples:
        lbl = e["label"]
        if isinstance(lbl, str):
            lbl = LABEL_MAP.get(lbl, 0)
        labels.append(int(lbl))
    counts  = np.bincount(labels, minlength=3).astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 3
    return torch.tensor(weights, dtype=torch.float).to(device)


# ── Train one model for one seed ──────────────────────────────────────────────
def train_one_seed(model_name, model_class, seed):
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} | seed={seed}")
    print(f"{'='*60}")

    set_seed(seed)

    model = model_class(num_labels=3)
    model.to(device)

    train_loader, cf_loader = build_loaders(model_name)
    class_weights = get_class_weights(device)
    ce_loss_fn    = torch.nn.CrossEntropyLoss(weight=class_weights)

    max_epochs   = {"bert": 10, "baseline": 35,
                    "ablation": 50, "proposed": 50}[model_name]
    warmup_steps = 200 if model_name == "bert" else 400
    total_steps  = len(train_loader) * max_epochs

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)

    cf_loss_fn       = CFContrastiveLoss(margin=1.0)
    boundary_loss_fn = BoundaryContrastiveLoss(margin=0.5)

    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0
    cf_iter     = iter(cf_loader) if cf_loader is not None else None

    for epoch in range(max_epochs):
        model.train()
        warmup_weight = min(1.0, (epoch + 1) / 10.0) \
                        if model_name == "proposed" else 1.0

        for batch in tqdm(train_loader,
                          desc=f"  Epoch {epoch+1}/{max_epochs}",
                          leave=False):
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_std     = batch["label"].to(device)

            output = model(input_ids, attention_mask)
            loss   = ce_loss_fn(output["logits"], labels_std)

            # CF batch for ablation and proposed
            if model_name in ("ablation", "proposed") and cf_iter is not None:
                try:
                    cf_batch = next(cf_iter)
                except StopIteration:
                    cf_iter  = iter(cf_loader)
                    cf_batch = next(cf_iter)

                orig_ids   = cf_batch["orig_input_ids"].to(device)
                orig_mask  = cf_batch["orig_attention_mask"].to(device)
                cf_ids     = cf_batch["cf_input_ids"].to(device)
                cf_mask    = cf_batch["cf_attention_mask"].to(device)
                orig_labels = cf_batch["orig_label"].to(device)

                orig_out = model(orig_ids, orig_mask)
                cf_out   = model(cf_ids,   cf_mask)

                loss += ce_loss_fn(orig_out["logits"], orig_labels)

                if model_name == "proposed":
                    orig_emb = orig_out["embeddings"]
                    cf_emb   = cf_out["embeddings"]

                    # Pairwise contrastive loss
                    loss += 0.3 * warmup_weight * cf_loss_fn(orig_emb, cf_emb)

                    # Boundary contrastive loss
                    emb       = output["embeddings"]
                    off_embs  = emb[labels_std == 1]
                    hate_embs = emb[labels_std == 2]
                    if off_embs.shape[0] > 0 and hate_embs.shape[0] > 0:
                        # Compute boundary loss manually without calling BoundaryContrastiveLoss
                        # to avoid the internal masking conflict
                        off_mean  = off_embs.mean(dim=0, keepdim=True)
                        hate_mean = hate_embs.mean(dim=0, keepdim=True)
                        cos_sim   = torch.nn.functional.cosine_similarity(
                            off_mean, hate_mean, dim=1)
                        boundary_loss = torch.relu(cos_sim + 0.5).mean()
                        loss += 0.1 * warmup_weight * boundary_loss

                    # Rationale supervision loss
                    rat_mask = cf_batch["orig_rationale_mask"].to(device)
                    has_rat  = (rat_mask.sum(dim=1) > 0).float()
                    if has_rat.sum() > 0:
                        probs = torch.softmax(orig_out["logits"], dim=1)
                        conf  = probs.max(dim=1).values
                        loss += 0.1 * (has_rat * (1 - conf)).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        val_f1 = evaluate_val_macro_f1(model, device, model_name)
        print(f"  Epoch {epoch+1:02d} | val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
            no_improve  = 0
            print(f"    New best: {best_val_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best and evaluate on test
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    preds, labels = get_predictions(model, "test", device, model_name)
    metrics = compute_metrics(preds, labels)
    metrics["best_val_f1"] = float(best_val_f1)
    print(f"  Test macro F1: {metrics['macro_f1']:.4f}")

    del model
    torch.cuda.empty_cache()
    return metrics


# ── Run all seeds for all models ──────────────────────────────────────────────
model_configs = [
    ("bert",     BERTBaseline),
    ("baseline", HateBERTBaseline),
    ("ablation", AblationCFOnlyModel),
    ("proposed", ProposedModel),
]

import shutil

all_results = {}
results_path = Path(config["paths"]["results"]) / "multiseed_results.json"

# Resume from partial results if session died previously
partial_path = Path("/content/drive/MyDrive/thesis_v3/multiseed_results_partial.json")
if partial_path.exists():
    print("Found partial results from previous session, resuming...")
    with open(partial_path) as f:
        all_results = json.load(f).get("per_seed", {})
    for model_name, runs in all_results.items():
        for run in runs:
            print(f"  Already done: {model_name} seed={run['seed']} "
                  f"macro_f1={run['metrics']['macro_f1']:.4f}")
else:
    print("No partial results found, starting fresh.")

for model_name, model_class in model_configs:
    if model_name not in all_results:
        all_results[model_name] = []

    completed_seeds = [r["seed"] for r in all_results[model_name]]

    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"  Skipping {model_name} seed={seed} (already done)")
            continue

        metrics = train_one_seed(model_name, model_class, seed)
        all_results[model_name].append({"seed": seed, "metrics": metrics})

        # Save after every single run
        with open(results_path, "w") as f:
            json.dump({"per_seed": all_results}, f, indent=2)
        shutil.copy(
            str(results_path),
            "/content/drive/MyDrive/thesis_v3/multiseed_results_partial.json")
        print(f"  Progress saved to Drive after {model_name} seed={seed}")

# ── Aggregate mean and std ────────────────────────────────────────────────────
metric_keys = ["accuracy", "macro_f1", "normal_f1", "offensive_f1",
               "hatespeech_f1", "harmful_subset_f1", "binary_f1"]

aggregated = {}
for model_name in ["bert", "baseline", "ablation", "proposed"]:
    runs = all_results[model_name]
    agg  = {}
    for k in metric_keys:
        vals   = [r["metrics"][k] for r in runs]
        agg[k] = {"mean": float(np.mean(vals)),
                  "std":  float(np.std(vals))}
    aggregated[model_name] = agg

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  MULTI-SEED RESULTS (mean +/- std over {len(SEEDS)} seeds)")
print(f"{'='*80}")
print(f"  {'Metric':<25} {'BERT':>18} {'Baseline':>18} "
      f"{'Ablation':>18} {'Proposed':>18}")
print(f"{'─'*80}")

for k in metric_keys:
    row = f"  {k:<25}"
    for mn in ["bert", "baseline", "ablation", "proposed"]:
        m   = aggregated[mn][k]["mean"]
        s   = aggregated[mn][k]["std"]
        row += f"  {m:.4f}+/-{s:.4f}"
    print(row)

print(f"{'='*80}")

# ── Save results ──────────────────────────────────────────────────────────────
results_path = Path(config["paths"]["results"]) / "multiseed_results.json"
with open(results_path, "w") as f:
    json.dump({"per_seed": all_results, "aggregated": aggregated}, f, indent=2)
print(f"\nSaved to {results_path}")

import shutil
shutil.copy(str(results_path),
            "/content/drive/MyDrive/thesis_v3/multiseed_results.json")
print("Saved to Drive")