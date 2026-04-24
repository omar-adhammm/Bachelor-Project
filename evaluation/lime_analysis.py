# evaluation/lime_analysis.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer

from configs.config_loader import load_config
from models.hatebert_baseline import HateBERTBaseline
from models.proposed_model    import ProposedModel
from models.ablation_cf_only  import AblationCFOnlyModel
from evaluation.metrics       import load_model_from_checkpoint

config     = load_config()
LABEL_NAMES = ["normal", "offensive", "hatespeech"]
device      = "cpu"


# ── Prediction function for LIME ──────────────────────────────────────────────

def make_predictor(model, tokenizer, device="cpu"):
    """
    Returns a prediction function compatible with LIME.
    LIME calls this with a list of text strings and expects
    a numpy array of shape [n_samples, n_classes].
    """
    def predictor(texts: list[str]) -> np.ndarray:
        model.eval()
        all_probs = []

        # Process in small batches to avoid memory issues
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoding = tokenizer(
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
                probs  = torch.softmax(output["logits"], dim=1)

            all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    return predictor


# ── Run LIME on one example ───────────────────────────────────────────────────

def explain_example(
    text:        str,
    predictor,
    true_label:  int,
    num_samples: int = 500,
    num_features: int = 10,
) -> dict:
    """
    Run LIME on a single text example.
    Returns explanation dict with token weights.
    """
    explainer = LimeTextExplainer(
        class_names=LABEL_NAMES,
        split_expression=r"\s+",
        bow=False,
    )

    explanation = explainer.explain_instance(
        text,
        predictor,
        num_features=num_features,
        num_samples=num_samples,
        labels=[0, 1, 2],
    )

    # Get predicted label
    probs       = predictor([text])[0]
    pred_label  = int(np.argmax(probs))

    # Extract token weights for the predicted class
    token_weights = dict(explanation.as_list(label=pred_label))

    return {
        "text":          text,
        "true_label":    LABEL_NAMES[true_label],
        "pred_label":    LABEL_NAMES[pred_label],
        "confidence":    float(probs[pred_label]),
        "probs":         probs.tolist(),
        "token_weights": token_weights,
        "explanation":   explanation,
    }


# ── Visualize token weights ───────────────────────────────────────────────────

def visualize_explanation(
    result:     dict,
    model_name: str,
    save_path:  str = None,
):
    """
    Create a colored token visualization showing which words
    pushed toward harmful vs normal classification.
    """
    text    = result["text"]
    weights = result["token_weights"]
    tokens  = text.split()

    # Normalize weights to [-1, 1]
    if weights:
        max_abs = max(abs(v) for v in weights.values()) or 1.0
        norm_weights = {k: v / max_abs for k, v in weights.items()}
    else:
        norm_weights = {}

    fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.8), 2.5))
    ax.axis("off")

    x, y    = 0.02, 0.5
    x_step  = 0.95 / max(len(tokens), 1)

    for token in tokens:
        weight = norm_weights.get(token, 0.0)

        # Color: red = harmful signal, green = normal signal
        if weight > 0:
            color = (1.0, 1.0 - weight * 0.7, 1.0 - weight * 0.7)  # red tint
        elif weight < 0:
            color = (1.0 + weight * 0.7, 1.0, 1.0 + weight * 0.7)  # green tint
        else:
            color = (0.95, 0.95, 0.95)  # gray

        ax.text(
            x, y, token,
            ha="left", va="center",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=color,
                edgecolor="gray",
                alpha=0.9,
            ),
            transform=ax.transAxes,
        )
        x += x_step + len(token) * 0.008

    # Title
    title = (
        f"{model_name} | "
        f"True: {result['true_label']} | "
        f"Pred: {result['pred_label']} ({result['confidence']:.0%})"
    )
    ax.set_title(title, fontsize=11, pad=10, fontweight="bold")

    # Legend
    red_patch   = mpatches.Patch(color=(1.0, 0.3, 0.3), label="Pushes toward harmful")
    green_patch = mpatches.Patch(color=(0.3, 1.0, 0.3), label="Pushes toward normal")
    ax.legend(
        handles=[red_patch, green_patch],
        loc="lower right",
        fontsize=9,
        framealpha=0.8,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ── Compare models on same examples ──────────────────────────────────────────

def compare_models_lime(
    models_dict: dict,
    examples:    list[dict],
    output_dir:  str,
    num_samples: int = 300,
):
    """
    Run LIME on the same examples across all models and
    save side-by-side visualizations.
    """
    tokenizer  = AutoTokenizer.from_pretrained(config["models"]["hatebert"]["name"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name, model in models_dict.items():
        print(f"\n── LIME Analysis: {model_name.upper()} ──")
        predictor = make_predictor(model, tokenizer, device)
        model_results = []

        for i, ex in enumerate(examples):
            print(f"  Example {i+1}/{len(examples)}: {ex['text'][:60]}...")
            result = explain_example(
                ex["text"],
                predictor,
                true_label=ex["label_id"],
                num_samples=num_samples,
            )
            model_results.append(result)

            # Save individual visualization
            save_path = output_dir / f"{model_name}_example_{i+1}.png"
            visualize_explanation(result, model_name, str(save_path))

            print(f"    True: {result['true_label']:12s} "
                  f"Pred: {result['pred_label']:12s} "
                  f"Conf: {result['confidence']:.2%}")

        all_results[model_name] = model_results

    # Save combined comparison figure
    _save_comparison_figure(all_results, examples, output_dir)

    # Save JSON results
    json_results = {}
    for model_name, results in all_results.items():
        json_results[model_name] = [
            {
                "text":          r["text"],
                "true_label":    r["true_label"],
                "pred_label":    r["pred_label"],
                "confidence":    r["confidence"],
                "token_weights": r["token_weights"],
            }
            for r in results
        ]

    with open(output_dir / "lime_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nAll LIME results saved to: {output_dir}")
    return all_results


def _save_comparison_figure(all_results, examples, output_dir):
    """Save a grid figure comparing all models on all examples."""
    model_names = list(all_results.keys())
    n_models    = len(model_names)
    n_examples  = len(examples)

    fig, axes = plt.subplots(
        n_examples, n_models,
        figsize=(7 * n_models, 2.5 * n_examples),
    )

    if n_examples == 1:
        axes = [axes]
    if n_models == 1:
        axes = [[ax] for ax in axes]

    for ex_i in range(n_examples):
        for m_i, model_name in enumerate(model_names):
            ax     = axes[ex_i][m_i]
            result = all_results[model_name][ex_i]
            tokens = result["text"].split()
            weights = result["token_weights"]

            ax.axis("off")

            if weights:
                max_abs = max(abs(v) for v in weights.values()) or 1.0
                norm_weights = {k: v / max_abs for k, v in weights.items()}
            else:
                norm_weights = {}

            x, y   = 0.01, 0.5
            x_step = 0.95 / max(len(tokens), 1)

            for token in tokens:
                w = norm_weights.get(token, 0.0)
                if w > 0:
                    color = (1.0, 1.0 - w * 0.7, 1.0 - w * 0.7)
                elif w < 0:
                    color = (1.0 + w * 0.7, 1.0, 1.0 + w * 0.7)
                else:
                    color = (0.95, 0.95, 0.95)

                ax.text(
                    x, y, token,
                    ha="left", va="center", fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=color,
                        edgecolor="gray",
                        alpha=0.85,
                    ),
                    transform=ax.transAxes,
                )
                x += x_step + len(token) * 0.006

            title = (
                f"{model_name}\n"
                f"Pred: {result['pred_label']} ({result['confidence']:.0%})"
            )
            ax.set_title(title, fontsize=9, pad=6)

    plt.suptitle(
        "LIME Token Attribution Comparison Across Models\n"
        "Red = pushes toward harmful | Green = pushes toward normal",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    save_path = output_dir / "lime_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison figure: {save_path}")


# ── Select interesting test examples ─────────────────────────────────────────

def select_examples(n_per_class: int = 2) -> list[dict]:
    """
    Select interesting examples from the test set for LIME analysis.
    Picks examples where models might differ — implicit harmful content.
    """
    from training.data_loader import load_split
    examples = load_split("test")

    selected = []
    label_map = {"normal": 0, "offensive": 1, "hatespeech": 2}

    # Pick short-to-medium length examples (easier to visualize)
    for label_name in ["offensive", "hatespeech", "normal"]:
        candidates = [
            ex for ex in examples
            if ex["label"] == label_name
            and 5 <= len(ex["text"].split()) <= 20
        ]
        # Sort by text length for cleaner visualization
        candidates = sorted(candidates, key=lambda x: len(x["text"].split()))
        selected.extend(candidates[:n_per_class])

    print(f"Selected {len(selected)} examples for LIME analysis:")
    for ex in selected:
        print(f"  [{ex['label']:12s}] {ex['text'][:70]}...")

    return selected

# ── Select Implicit test examples ─────────────────────────────────────────

def select_implicit_examples(n_per_class: int = 2) -> list[dict]:
    """
    Select implicit harmful examples — harmful texts WITHOUT obvious slur words.
    These are harder cases where model differences are more visible.
    """
    from training.data_loader import load_split

    # Common explicit slurs to avoid
    explicit_words = {
        "nigger", "nigga", "faggot", "fag", "chink", "spic", "kike",
        "retard", "retarded", "jihadist", "jihadi", "tranny", "dyke",
        "cunt", "bitch", "whore", "slut"
    }

    examples   = load_split("test")
    selected   = []

    for label_name in ["offensive", "hatespeech", "normal"]:
        candidates = []
        for ex in examples:
            if ex["label"] != label_name:
                continue
            tokens     = ex["text"].lower().split()
            has_slur   = any(t in explicit_words for t in tokens)
            word_count = len(tokens)

            # Want implicit examples: no slurs, medium length
            if not has_slur and 8 <= word_count <= 25:
                candidates.append(ex)

        # Sort by length for cleaner visualization
        candidates = sorted(candidates, key=lambda x: len(x["text"].split()))
        selected.extend(candidates[:n_per_class])

    print(f"Selected {len(selected)} implicit examples for LIME analysis:")
    for ex in selected:
        print(f"  [{ex['label']:12s}] {ex['text'][:70]}...")

    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    print("=== LIME Attribution Analysis ===\n")

    output_dir     = Path(config["paths"]["results"]) / "lime_analysis"
    checkpoint_dir = Path(config["paths"]["checkpoints"])

    def find_best_checkpoint(prefix):
        ckpts = list(checkpoint_dir.glob(f"{prefix}_epoch_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found for {prefix}")
        return str(min(ckpts, key=lambda p: float(p.stem.split("loss_")[1])))

    # Load all three models
    print("Loading models...\n")
    models_dict = {
        "baseline": load_model_from_checkpoint(
            HateBERTBaseline, find_best_checkpoint("baseline"), device
        ),
        "ablation": load_model_from_checkpoint(
            AblationCFOnlyModel, find_best_checkpoint("ablation"), device
        ),
        "proposed": load_model_from_checkpoint(
            ProposedModel, find_best_checkpoint("proposed"), device
        ),
    }

    # Select examples
    print("\nSelecting test examples...\n")
    examples = select_implicit_examples(n_per_class=2)

    # Run LIME
    print("\nRunning LIME analysis (this takes ~10-15 minutes)...\n")
    results = compare_models_lime(
        models_dict,
        examples,
        str(output_dir),
        num_samples=300,
    )

    print("\nLIME analysis complete!")
    print(f"Visualizations saved to: {output_dir}")
    print("\nFiles generated:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")