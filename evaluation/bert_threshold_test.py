import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from datasets import load_dataset

from configs.config_loader import load_config
from models.bert_baseline import BERTBaseline

config = load_config()
device = "cpu"

class ToxiGenDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex  = self.examples[idx]
        # Clean text for BERT compatibility
        text = ex["text"].encode("ascii", errors="ignore").decode("ascii")
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(ex["label"], dtype=torch.long),
        }

def load_toxigen():
    dataset  = load_dataset("toxigen/toxigen-data",
                            name="annotated", split="test")
    examples = []
    for row in dataset:
        toxicity = row.get("toxicity_human", 0)
        if toxicity is None:
            continue
        label = 1 if float(toxicity) >= 3.5 else 0
        examples.append({"text": row["text"], "label": label})
    print(f"Loaded {len(examples)} examples")
    return examples

def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model = BERTBaseline(num_labels=3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    examples  = load_toxigen()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ckpt_path = (Path(config["paths"]["checkpoints"]) /
                 "bert_epoch_2_loss_0.8036.pt")
    model     = load_model(ckpt_path, device)

    dataset = ToxiGenDataset(examples, tokenizer, max_length=128)
    loader  = DataLoader(dataset, batch_size=32,
                         shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Getting probabilities"):
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device))
            probs = torch.softmax(output["logits"], dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    p_harmful  = all_probs[:, 1] + all_probs[:, 2]

    print(f"\n{'Strategy':<45} {'Toxic F1':>10} {'Macro F1':>10} {'Accuracy':>10}")
    print(f"{'─'*77}")

    preds = (np.argmax(all_probs, axis=1) != 0).astype(int)
    f1    = f1_score(all_labels, preds, average="binary", zero_division=0)
    mac   = f1_score(all_labels, preds, average="macro",  zero_division=0)
    acc   = accuracy_score(all_labels, preds)
    print(f"{'Original (argmax)':<45} {f1:>10.4f} {mac:>10.4f} {acc:>10.4f}")

    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        preds = (p_harmful >= threshold).astype(int)
        f1    = f1_score(all_labels, preds, average="binary", zero_division=0)
        mac   = f1_score(all_labels, preds, average="macro",  zero_division=0)
        acc   = accuracy_score(all_labels, preds)
        print(f"{'p(off+hate) >= ' + str(threshold):<45} "
              f"{f1:>10.4f} {mac:>10.4f} {acc:>10.4f}")