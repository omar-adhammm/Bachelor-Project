# evaluation/compare_models.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

from configs.config_loader import load_config

config = load_config()


# ── Load results ──────────────────────────────────────────────────────────────

def load_results(results_path: str = None) -> dict:
    """Load final evaluation results from JSON."""
    if results_path is None:
        results_path = Path(config["paths"]["results"]) / "final_evaluation.json"

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded results for: {list(results.keys())}")
    return results


# ── Text comparison table ─────────────────────────────────────────────────────

def print_comparison_table(results: dict):
    """Print a formatted comparison table to terminal."""
    models = ["baseline", "ablation", "proposed"]
    metrics = [
        ("Accuracy",           "accuracy"),
        ("Macro F1",           "macro_f1"),
        ("Weighted F1",        "weighted_f1"),
        ("Normal F1",          "normal_f1"),
        ("Offensive F1",       "offensive_f1"),
        ("Hate Speech F1",     "hatespeech_f1"),
        ("Harmful-subset F1",  "harmful_subset_f1"),
        ("Binary F1",          "binary_f1"),
    ]

    col_w = 22
    print("\n" + "=" * 80)
    print("  FINAL MODEL COMPARISON — TEST SET")
    print("=" * 80)
    header = f"  {'Metric':<24}"
    for m in models:
        header += f"  {m.capitalize():>10}"
    header += "  Delta (B→P)"
    print(header)
    print("─" * 80)

    for display, key in metrics:
        vals = {m: results[m][key] for m in models if m in results}
        row  = f"  {display:<24}"

        best = max(vals.values())
        for m in models:
            val    = vals.get(m, 0)
            marker = "*" if abs(val - best) < 1e-6 and len(vals) > 1 else " "
            row   += f"  {val:.4f}{marker}"

        # Delta: proposed vs baseline
        if "baseline" in vals and "proposed" in vals:
            delta = vals["proposed"] - vals["baseline"]
            sign  = "+" if delta >= 0 else ""
            row  += f"  {sign}{delta:.4f}"

        if key == "harmful_subset_f1":
            row += "  ← KEY"

        print(row)

    print("─" * 80)
    print("  * = best value for this metric")

    # Improvement summary
    if "baseline" in results and "proposed" in results:
        delta_macro    = results["proposed"]["macro_f1"]        - results["baseline"]["macro_f1"]
        delta_harmful  = results["proposed"]["harmful_subset_f1"] - results["baseline"]["harmful_subset_f1"]
        delta_offensive= results["proposed"]["offensive_f1"]    - results["baseline"]["offensive_f1"]
        delta_hate     = results["proposed"]["hatespeech_f1"]   - results["baseline"]["hatespeech_f1"]

        print("\n  PROPOSED vs BASELINE improvements:")
        print(f"    Macro F1:          {'+' if delta_macro>=0 else ''}{delta_macro:.4f}")
        print(f"    Harmful-subset F1: {'+' if delta_harmful>=0 else ''}{delta_harmful:.4f}")
        print(f"    Offensive F1:      {'+' if delta_offensive>=0 else ''}{delta_offensive:.4f}")
        print(f"    Hate Speech F1:    {'+' if delta_hate>=0 else ''}{delta_hate:.4f}")

    print("=" * 80 + "\n")


# ── Bar chart ─────────────────────────────────────────────────────────────────

def plot_metric_bars(results: dict, save_path: str = None):
    """Bar chart comparing all models across key metrics."""
    models  = ["baseline", "ablation", "proposed"]
    colors  = ["#566573", "#148F77", "#0D1B2A"]
    labels  = ["Baseline", "Ablation\n(CF only)", "Proposed\n(CF + Contrastive)"]

    metrics = [
        ("Accuracy",          "accuracy"),
        ("Macro F1",          "macro_f1"),
        ("Normal F1",         "normal_f1"),
        ("Offensive F1",      "offensive_f1"),
        ("Hate Speech F1",    "hatespeech_f1"),
        ("Harmful-subset F1", "harmful_subset_f1"),
    ]

    n_metrics = len(metrics)
    n_models  = len(models)
    x         = np.arange(n_metrics)
    width     = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (model, color, label) in enumerate(zip(models, colors, labels)):
        vals = [results[model][key] for _, key in metrics]
        bars = ax.bar(
            x + i * width, vals, width,
            label=label, color=color, alpha=0.87,
            edgecolor="white", linewidth=0.5,
        )
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, color="#2C3E50",
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Model Comparison — Test Set Performance\n"
        "HateXplain 3-class Classification (Normal / Offensive / Hate Speech)",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([m for m, _ in metrics], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Highlight harmful-subset bar group
    ax.axvspan(4.6, 5.85, alpha=0.08, color="#E74C3C", zorder=0)
    ax.text(5.22, 0.95, "KEY\nMETRIC", ha="center", fontsize=8,
            color="#E74C3C", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved bar chart: {save_path}")
    plt.close()


# ── Per-class F1 radar chart ──────────────────────────────────────────────────

def plot_radar_chart(results: dict, save_path: str = None):
    """Radar chart showing per-class F1 for each model."""
    categories = ["Normal F1", "Offensive F1", "Hate Speech F1",
                  "Macro F1",  "Binary F1"]
    keys       = ["normal_f1", "offensive_f1", "hatespeech_f1",
                  "macro_f1",  "binary_f1"]

    models  = ["baseline", "ablation", "proposed"]
    colors  = ["#566573",  "#148F77",  "#0D1B2A"]
    labels  = ["Baseline", "Ablation (CF only)", "Proposed (CF + Contrastive)"]

    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for model, color, label in zip(models, colors, labels):
        vals = [results[model][k] for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title(
        "Per-Class F1 Comparison\n(Radar Chart)",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.grid(color="gray", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved radar chart: {save_path}")
    plt.close()


# ── Confusion matrix heatmap ──────────────────────────────────────────────────

def plot_confusion_matrices(results: dict, save_path: str = None):
    """Side-by-side confusion matrix heatmaps for all models."""
    models = ["baseline", "ablation", "proposed"]
    titles = ["Baseline", "Ablation (CF only)", "Proposed (CF + Contrastive)"]
    labels = ["Normal", "Offensive", "Hate"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, model, title in zip(axes, models, titles):
        cm = np.array(results[model]["confusion_matrix"])

        # Normalize by row (true label)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest",
                       cmap="Blues", vmin=0, vmax=1)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)

        # Annotate cells with count and percentage
        for i in range(3):
            for j in range(3):
                count   = cm[i][j]
                pct     = cm_norm[i][j]
                color   = "white" if pct > 0.5 else "#1A252F"
                ax.text(j, i, f"{count}\n({pct:.0%})",
                        ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "Confusion Matrices — Test Set (normalized by true label)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrices: {save_path}")
    plt.close()


# ── Training history plot ─────────────────────────────────────────────────────

def plot_training_history(save_path: str = None):
    """Plot training loss curves from saved JSON results files."""
    results_dir = Path(config["paths"]["results"])

    # Map each model to its specific results file
    model_files = {
        "baseline": "training_results_20260418_023009.json",
        "proposed": "training_results_20260418_104021.json",
        "ablation": "training_results_20260418_203909.json",
    }

    colors = {"baseline": "#566573", "proposed": "#0D1B2A", "ablation": "#148F77"}
    labels = {"baseline": "Baseline", "proposed": "Proposed", "ablation": "Ablation"}

    history = {}
    for model_name, filename in model_files.items():
        filepath = results_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping {model_name}")
            continue
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        model_data = data.get("models", {}).get(model_name, {})
        hist = model_data.get("history", {})
        if hist.get("train_loss"):
            history[model_name] = hist
            print(f"Loaded {model_name} history: {len(hist['train_loss'])} epochs")

    if not history:
        print("No training history found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for model_name, hist in history.items():
        color  = colors[model_name]
        label  = labels[model_name]
        epochs = range(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs, hist["train_loss"], "o-",
                     color=color, label=label, linewidth=2, markersize=6)
        axes[1].plot(epochs, hist["val_loss"], "o-",
                     color=color, label=label, linewidth=2, markersize=6)

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss",  fontsize=11)
        ax.set_title(title,    fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.suptitle("Training History — All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training history: {save_path}")
    plt.close()


# ── Save summary JSON ─────────────────────────────────────────────────────────

def save_summary(results: dict, save_path: str = None):
    """Save a clean human-readable summary JSON."""
    if save_path is None:
        save_path = Path(config["paths"]["results"]) / "comparison_summary.json"

    summary = {
        "models": {},
        "best_per_metric": {},
        "proposed_vs_baseline": {},
    }

    metrics = [
        "accuracy", "macro_f1", "weighted_f1",
        "normal_f1", "offensive_f1", "hatespeech_f1",
        "harmful_subset_f1", "binary_f1",
    ]

    for model in results:
        summary["models"][model] = {k: round(results[model][k], 4) for k in metrics}

    for metric in metrics:
        vals = {m: results[m][metric] for m in results}
        best = max(vals, key=vals.get)
        summary["best_per_metric"][metric] = {
            "best_model": best,
            "value":      round(vals[best], 4),
        }

    if "baseline" in results and "proposed" in results:
        for metric in metrics:
            delta = results["proposed"][metric] - results["baseline"][metric]
            summary["proposed_vs_baseline"][metric] = round(delta, 4)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Model Comparison ===\n")

    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results()

    # Print comparison table
    print_comparison_table(results)

    # Generate all plots
    print("Generating plots...\n")

    plot_metric_bars(
        results,
        save_path=str(results_dir / "comparison_bars.png"),
    )
    plot_radar_chart(
        results,
        save_path=str(results_dir / "comparison_radar.png"),
    )
    plot_confusion_matrices(
        results,
        save_path=str(results_dir / "confusion_matrices.png"),
    )
    plot_training_history(
        save_path=str(results_dir / "training_history.png"),
    )

    # Save summary
    save_summary(results)

    print("\nAll outputs saved to:", results_dir)
    print("\nFiles generated:")
    for f in sorted(results_dir.glob("*.png")):
        print(f"  {f.name}")
    print("\ncompare_models.py complete!")