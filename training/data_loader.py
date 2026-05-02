import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from configs.config_loader import load_config

config = load_config()

# ── 1. Raw JSON loader ────────────────────────────────────────────────────────

def load_split(split: str) -> list[dict]:
    """Load train / validation / test from data/raw/"""
    path = f"{config['paths']['raw_data']}/{split}.json"
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {path}")
    return data


# ── 2. PyTorch Dataset ────────────────────────────────────────────────────────

class HateXplainDataset(Dataset):
    """
    Standard dataset — one example per row.
    Used for: HateBERT baseline, zero-shot Llama, ablation.
    """

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 128):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex       = self.examples[idx]
        encoding = self.tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Build token-level rationale mask aligned to tokenizer output
        # Map word-level rationale mask to subword tokens
        rationale_mask = self._align_rationale_mask(
            ex["text"],
            ex.get("rationale_mask", []),
            encoding,
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "rationale_mask": rationale_mask,
            "label":          torch.tensor(ex["label_id"], dtype=torch.long),
            "text":           ex["text"],
            "id":             ex["id"],
        }

    def _align_rationale_mask(
        self,
        text:          str,
        word_rationale: list,
        encoding,
    ) -> torch.Tensor:
        """
        Align word-level rationale mask to subword tokenizer output.
        Returns a tensor of shape [max_length] with 1 for rationale tokens.
        """
        mask = torch.zeros(self.max_length, dtype=torch.float)

        if not word_rationale or not any(word_rationale):
            return mask

        words = text.split()
        if len(word_rationale) != len(words):
            return mask  # mismatch — return empty mask

        # Use word_ids to map words to subword tokens
        try:
            word_ids = encoding.word_ids(batch_index=0)
        except Exception:
            return mask

        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if token_idx >= self.max_length:
                break
            if word_idx < len(word_rationale) and word_rationale[word_idx] == 1:
                mask[token_idx] = 1.0

        return mask


class ContrastiveHateDataset(Dataset):
    """
    Paired dataset — (original, counterfactual) per row.
    Used for: proposed model with contrastive loss.

    Each CF pair file (JSONL) must have:
        {
          "original":   { "text": ..., "label_id": 1 or 2 },
          "counterfactual": { "text": ..., "label_id": 0 }
        }
    """

    def __init__(
        self,
        pairs:      list[dict],
        tokenizer,
        max_length: int = 128,
        rationale_lookup: dict = None,  # NEW: id -> rationale_mask
    ):
        self.pairs            = pairs
        self.tokenizer        = tokenizer
        self.max_length       = max_length
        self.rationale_lookup = rationale_lookup or {}

    def __len__(self):
        return len(self.pairs)

    def _encode(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        orig = self._encode(pair["original"]["text"])
        cf   = self._encode(pair["counterfactual"]["text"])

        # Look up rationale mask for original
        orig_id        = pair["original"].get("id", "")
        word_rationale = self.rationale_lookup.get(orig_id, [])

        orig_rat_mask = torch.zeros(self.max_length, dtype=torch.float)
        if word_rationale:
            encoding = self.tokenizer(
                pair["original"]["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            try:
                word_ids = encoding.word_ids(batch_index=0)
                words    = pair["original"]["text"].split()
                for token_idx, word_idx in enumerate(word_ids):
                    if word_idx is None or token_idx >= self.max_length:
                        continue
                    if word_idx < len(word_rationale) and word_rationale[word_idx] == 1:
                        orig_rat_mask[token_idx] = 1.0
            except Exception:
                pass

        return {
            "orig_input_ids":      orig["input_ids"],
            "orig_attention_mask": orig["attention_mask"],
            "orig_rationale_mask": orig_rat_mask,
            "orig_label":          torch.tensor(pair["original"]["label_id"], dtype=torch.long),
            "cf_input_ids":        cf["input_ids"],
            "cf_attention_mask":   cf["attention_mask"],
            "cf_label":            torch.tensor(pair["counterfactual"]["label_id"], dtype=torch.long),
        }

# ── 3. CF pair loader ─────────────────────────────────────────────────────────

def load_cf_pairs(path: str) -> list[dict]:
    """Load generated counterfactual pairs from a JSONL file."""
    import jsonlines
    pairs = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            pairs.append(obj)
    print(f"Loaded {len(pairs)} CF pairs from {path}")
    return pairs


# ── 4. DataLoader factory ─────────────────────────────────────────────────────

def get_dataloaders(
    tokenizer,
    batch_size:  int = 16,
    max_length:  int = 128,
    splits: list[str] = ["train", "validation", "test"],
) -> dict[str, DataLoader]:
    """Return a dict of DataLoaders for the requested splits."""
    loaders = {}
    for split in splits:
        examples = load_split(split)
        dataset  = HateXplainDataset(examples, tokenizer, max_length)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,          # keep 0 on Windows to avoid multiprocessing issues
        )
        print(f"  {split} → {len(dataset)} examples, "
              f"{len(loaders[split])} batches of {batch_size}")
    return loaders


def get_contrastive_dataloader(
    cf_path:    str,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 128,
) -> DataLoader:
    """Return a DataLoader of (original, CF) pairs for contrastive training."""
    pairs   = load_cf_pairs(cf_path)
    dataset = ContrastiveHateDataset(pairs, tokenizer, max_length)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f"Contrastive loader → {len(dataset)} pairs, "
          f"{len(loader)} batches of {batch_size}")
    return loader


# ── 5. Quick smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_name = config["models"]["hatebert"]["name"]
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\n── Standard DataLoaders ──")
    loaders = get_dataloaders(
        tokenizer,
        batch_size=config["models"]["hatebert"]["batch_size"],
        max_length=config["models"]["hatebert"]["max_length"],
    )

    print("\n── Inspecting one train batch ──")
    batch = next(iter(loaders["train"]))
    print(f"  input_ids shape:      {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels:               {batch['label']}")
    print(f"  first text:           {batch['text'][0][:80]}...")

    print("\ndata_loader.py smoke test passed!")