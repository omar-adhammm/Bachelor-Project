import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from configs.config_loader import load_config
from training.data_loader import load_split
from models.bert_baseline import BERTBaseline
from models.hatebert_baseline import HateBERTBaseline
from models.proposed_model import ProposedModel
from models.ablation_cf_only import AblationCFOnlyModel

config     = load_config()
device     = "cpu"  # runs on laptop CPU
LABEL_MAP  = {0: "normal", 1: "offensive", 2: "hatespeech"}
LABEL_NAMES = ["normal", "offensive", "hatespeech"]
NUM_LIME_SAMPLES = 300
TOP_K_TOKENS     = 10


# ── Load model from checkpoint ────────────────────────────────────────────────
def load_model(model_class, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model = model_class(num_labels=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded: {checkpoint_path}")
    return model


# ── Build predictor function for LIME ────────────────────────────────────────
def make_predictor(model, tokenizer, device="cpu"):
    def predictor(texts):
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt")
        input_ids      = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            probs  = torch.softmax(output["logits"], dim=1)
        return probs.cpu().numpy()
    return predictor


# ── Compute IOU between two sets of tokens ────────────────────────────────────
def compute_iou(pred_tokens: set, true_tokens: set) -> float:
    if len(true_tokens) == 0:
        return 0.0
    intersection = pred_tokens & true_tokens
    union        = pred_tokens | true_tokens
    return len(intersection) / len(union) if len(union) > 0 else 0.0


# ── Get top-K tokens from LIME explanation ────────────────────────────────────
def get_lime_top_tokens(explanation, label, top_k=10) -> set:
    exp_list = explanation.as_list(label=label)
    # Sort by absolute importance, take top K positive contributions
    positive = [(token, weight) for token, weight in exp_list if weight > 0]
    positive_sorted = sorted(positive, key=lambda x: x[1], reverse=True)
    top_tokens = {token.lower() for token, _ in positive_sorted[:top_k]}
    return top_tokens


# ── Run LIME on examples with rationale annotations ───────────────────────────
def run_lime_iou(model, tokenizer, examples, model_name,
                 num_samples=300, top_k=10):
    explainer = LimeTextExplainer(class_names=LABEL_NAMES)
    predictor = make_predictor(model, tokenizer, device)

    iou_scores  = []
    all_examples = []

    # Filter to examples that have rationale annotations and are harmful
    annotated = [
        ex for ex in examples
        if ex.get("rationale_mask") and sum(ex["rationale_mask"]) > 0
        and ex.get("label_id", 0) in (1, 2)
    ]

    print(f"  Found {len(annotated)} examples with rationale annotations")
    print(f"  Running LIME on all {len(annotated)} examples...")

    for ex in tqdm(annotated, desc=f"  LIME [{model_name}]"):
        text   = ex["text"]
        label  = ex["label_id"]
        words  = text.split()

        # Get rationale tokens from human annotations
        rationale = ex["rationale_mask"]
        true_rationale_tokens = set()
        for i, flag in enumerate(rationale):
            if flag == 1 and i < len(words):
                true_rationale_tokens.add(words[i].lower())

        if len(true_rationale_tokens) == 0:
            continue

        # Run LIME
        try:
            explanation = explainer.explain_instance(
                text,
                predictor,
                labels=[label],
                num_features=top_k,
                num_samples=num_samples)

            pred_tokens = get_lime_top_tokens(explanation, label, top_k)
            iou = compute_iou(pred_tokens, true_rationale_tokens)
            iou_scores.append(iou)

            all_examples.append({
                "text":                  text,
                "label":                 label,
                "true_rationale_tokens": list(true_rationale_tokens),
                "pred_lime_tokens":      list(pred_tokens),
                "iou":                   iou,
                "explanation":           explanation.as_list(label=label),
            })
        except Exception as e:
            print(f"  Skipping example due to error: {e}")
            continue

    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    std_iou  = float(np.std(iou_scores))  if iou_scores else 0.0
    print(f"  Mean IOU F1: {mean_iou:.4f} +/- {std_iou:.4f}")

    return mean_iou, std_iou, all_examples


# ── Plot LIME explanation for a single example ────────────────────────────────
def plot_lime_example(example, model_name, example_idx, save_dir):
    text   = example["text"]
    label  = example["label"]
    exp    = example["explanation"]
    iou    = example["iou"]
    true_t = set(example["true_rationale_tokens"])
    pred_t = set(example["pred_lime_tokens"])

    words   = [e[0] for e in exp]
    weights = [e[1] for e in exp]

    colors = []
    for word in words:
        if word.lower() in true_t and word.lower() in pred_t:
            colors.append("#2ca02c")   # green: correct
        elif word.lower() in pred_t:
            colors.append("#1f77b4")   # blue: predicted but not true
        elif word.lower() in true_t:
            colors.append("#d62728")   # red: true but missed
        else:
            colors.append("#aec7e8")   # light blue: not highlighted

    fig, ax = plt.subplots(figsize=(10, 4))
    y_pos   = np.arange(len(words))
    ax.barh(y_pos, weights, color=colors, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=10)
    ax.set_xlabel("LIME importance weight", fontsize=11)
    ax.set_title(
        f"{model_name} | Label: {LABEL_MAP[label]} | IOU: {iou:.3f}\n"
        f"Text: {text[:80]}{'...' if len(text) > 80 else ''}",
        fontsize=10, wrap=True)

    patches = [
        mpatches.Patch(color="#2ca02c", label="Correct (LIME + Human)"),
        mpatches.Patch(color="#1f77b4", label="LIME only"),
        mpatches.Patch(color="#d62728", label="Human only (missed)"),
        mpatches.Patch(color="#aec7e8", label="Not highlighted"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    ax.axvline(x=0, color="black", linewidth=0.8)

    plt.tight_layout()
    save_path = save_dir / f"{model_name}_example_{example_idx+1}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Install lime if not installed
    try:
        import lime
    except ImportError:
        print("Installing lime...")
        os.system("pip install lime")
        import lime
        from lime.lime_text import LimeTextExplainer

    save_dir = Path("outputs/results/lime_iou_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config["paths"]["checkpoints"])
    test_examples  = load_split("test")
    #test_examples = test_examples[:50]  # test run with 50 examples first

    # Model configs
    model_configs = [
        ("proposed", ProposedModel, config["models"]["hatebert"]["name"],
         checkpoint_dir / "proposed_epoch_31_loss_0.6994.pt"),
    ]

    all_iou_results = {}

    # Load already completed results
    for mn in ["bert", "baseline", "ablation"]:
        result_path = save_dir / f"{mn}_lime_results.json"
        if result_path.exists():
            with open(result_path) as f:
                data = json.load(f)
            all_iou_results[mn] = {
                "mean_iou":   data["mean_iou"],
                "std_iou":    data["std_iou"],
                "n_examples": data["n_examples"],
            }
            print(f"Loaded existing results for {mn}: IOU={data['mean_iou']:.4f}")

    for model_name, model_class, tokenizer_name, ckpt_path in model_configs:
        print(f"\n{'='*60}")
        print(f"  {model_name.upper()}")
        print(f"{'='*60}")

        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}, skipping.")
            continue

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model     = load_model(model_class, ckpt_path, device)

        mean_iou, std_iou, examples = run_lime_iou(
            model, tokenizer, test_examples, model_name,
            num_samples=NUM_LIME_SAMPLES,
            top_k=TOP_K_TOKENS)

        all_iou_results[model_name] = {
            "mean_iou": mean_iou,
            "std_iou":  std_iou,
            "n_examples": len(examples),
        }

        # Save top 5 examples by IOU for visualization
        top_examples = sorted(examples, key=lambda x: x["iou"], reverse=True)[:3]
        low_examples = sorted(examples, key=lambda x: x["iou"])[:2]
        selected     = top_examples + low_examples

        for i, ex in enumerate(selected):
            plot_lime_example(ex, model_name, i, save_dir)

        # Save all example data
        with open(save_dir / f"{model_name}_lime_results.json", "w") as f:
            json.dump({
                "mean_iou":   mean_iou,
                "std_iou":    std_iou,
                "n_examples": len(examples),
                "examples":   examples,
            }, f, indent=2)

        del model
        print(f"  Done. Mean IOU: {mean_iou:.4f}")

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  IOU F1 COMPARISON (LIME vs Human Rationales)")
    print(f"{'='*60}")
    print(f"  {'Model':<15} {'Mean IOU':>12} {'Std IOU':>10} {'N Examples':>12}")
    print(f"{'─'*60}")
    for mn, res in all_iou_results.items():
        print(f"  {mn:<15} {res['mean_iou']:>12.4f} "
              f"{res['std_iou']:>10.4f} {res['n_examples']:>12}")
    print(f"{'='*60}")

    # ── Save summary ───────────────────────────────────────────────────────────
    with open(save_dir / "iou_summary.json", "w") as f:
        json.dump(all_iou_results, f, indent=2)
    print(f"\nAll results saved to {save_dir}")