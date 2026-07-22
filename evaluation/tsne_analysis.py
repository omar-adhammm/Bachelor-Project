import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
from tqdm import tqdm

from configs.config_loader import load_config
from training.data_loader import load_split
from models.bert_baseline import BERTBaseline
from models.proposed_model import ProposedModel
from models.hatebert_baseline import HateBERTBaseline
from models.ablation_cf_only import AblationCFOnlyModel

config = load_config()
device = "cpu"
LABEL_NAMES  = ["Normal", "Offensive", "Hate Speech"]
LABEL_COLORS = ["#2ca02c", "#1f77b4", "#d62728"]


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


# ── Extract embeddings ────────────────────────────────────────────────────────
def extract_embeddings(model, examples, tokenizer, device="cpu"):
    dataset = SimpleDataset(examples, tokenizer, max_length=128)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    all_embs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Extracting embeddings", leave=False):
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device))
            embs = output["embeddings"].cpu().numpy()
            all_embs.extend(embs)
            all_labels.extend(batch["label"].numpy())

    return np.array(all_embs), np.array(all_labels)


# ── Plot t-SNE ────────────────────────────────────────────────────────────────
def plot_tsne(embeddings, labels, title, save_path, perplexity=30):
    print(f"  Running t-SNE for {title}...")
    tsne   = TSNE(n_components=2, perplexity=perplexity,
                  random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))

    for class_idx, (name, color) in enumerate(zip(LABEL_NAMES, LABEL_COLORS)):
        mask = labels == class_idx
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=name, alpha=0.45,
            s=12, linewidths=0)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE dimension 2", fontsize=11)
    ax.legend(fontsize=11, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Plot side by side comparison ──────────────────────────────────────────────
def plot_comparison(all_coords, all_labels, model_names, save_path):
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, coords, labels, name in zip(
            axes, all_coords, all_labels, model_names):
        for class_idx, (cname, color) in enumerate(
                zip(LABEL_NAMES, LABEL_COLORS)):
            mask = labels == class_idx
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, label=cname, alpha=0.45,
                s=10, linewidths=0)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [mpatches.Patch(color=c, label=n)
               for c, n in zip(LABEL_COLORS, LABEL_NAMES)]
    fig.legend(handles=handles, fontsize=11,
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("t-SNE Embedding Space Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    save_dir = Path("outputs/results/tsne_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config["paths"]["checkpoints"])
    test_examples  = load_split("test")
    print(f"Test examples: {len(test_examples)}")

    model_configs = [
        ("BERT Baseline",
         BERTBaseline,
         "bert-base-uncased",
         checkpoint_dir / "bert_epoch_2_loss_0.8036.pt"),
        ("Proposed Model",
         ProposedModel,
         config["models"]["hatebert"]["name"],
         checkpoint_dir / "proposed_epoch_31_loss_0.6994.pt"),
         ("BERTweet Baseline",
         HateBERTBaseline,
         config["models"]["hatebert"]["name"],
         checkpoint_dir / "baseline_epoch_19_loss_0.7006.pt"),
        ("Ablation",
         AblationCFOnlyModel,
         config["models"]["hatebert"]["name"],
         checkpoint_dir / "ablation_epoch_27_loss_0.7102.pt"),
    ]

    all_coords = []
    all_labels = []
    model_names = []

    for display_name, model_class, tokenizer_name, ckpt_path in model_configs:
        print(f"\n{'='*55}")
        print(f"  {display_name}")
        print(f"{'='*55}")

        if not ckpt_path.exists():
            print(f"  Checkpoint not found: {ckpt_path}")
            continue

        tokenizer  = AutoTokenizer.from_pretrained(tokenizer_name)
        model      = load_model(model_class, ckpt_path, device)
        embs, lbls = extract_embeddings(model, test_examples, tokenizer, device)

        print(f"  Embeddings shape: {embs.shape}")
        print(f"  Running t-SNE...")

        tsne   = TSNE(n_components=2, perplexity=30,
                      random_state=42, max_iter=1000)
        coords = tsne.fit_transform(embs)

        all_coords.append(coords)
        all_labels.append(lbls)
        model_names.append(display_name)

        # Individual plot
        plot_tsne(embs, lbls,
                  f"t-SNE: {display_name}",
                  save_dir / f"tsne_{display_name.lower().replace(' ', '_')}.png")

        del model

    # Side by side comparison plot
    if len(all_coords) >= 2:
        plot_comparison(
            all_coords, all_labels, model_names,
            save_dir / "tsne_comparison.png")

    print("\nAll t-SNE plots saved to outputs/results/tsne_analysis/")