# models/hatebert_baseline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from configs.config_loader import load_config

config = load_config()


class HateBERTBaseline(nn.Module):
    """
    Model A — HateBERT fine-tuned on HateXplain only.
    No counterfactual data, no contrastive loss.
    This is the baseline everything else is compared against.
    """

    def __init__(self, num_labels: int = 3):
        super().__init__()
        model_name = config["models"]["hatebert"]["name"]

        print(f"Loading HateBERT: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        self.num_labels = num_labels

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         torch.Tensor = None,
    ) -> dict:
        """
        Args:
            input_ids:      [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels:         [batch_size] optional — if provided, loss is computed

        Returns:
            dict with keys: loss (optional), logits, embeddings
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # CLS token embedding from last hidden state
        # Shape: [batch_size, hidden_size]
        cls_embedding = outputs.hidden_states[-1][:, 0, :]

        result = {
            "logits":     outputs.logits,
            "embeddings": cls_embedding,
        }

        if labels is not None:
            result["loss"] = outputs.loss

        return result

    def get_embeddings(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract CLS embeddings without computing loss."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        return outputs.hidden_states[-1][:, 0, :]


def get_tokenizer():
    """Return the HateBERT tokenizer."""
    return AutoTokenizer.from_pretrained(config["models"]["hatebert"]["name"])


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== HateBERT Baseline Smoke Test ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model and tokenizer
    tokenizer = get_tokenizer()
    model     = HateBERTBaseline(num_labels=3).to(device)

    # Sample texts
    texts = [
        "I hate all immigrants they should be deported.",
        "That movie was absolutely terrible.",
        "I enjoyed spending time with my friends today.",
    ]
    labels = torch.tensor([2, 1, 0]).to(device)  # hate, offensive, normal

    # Tokenize
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config["models"]["hatebert"]["max_length"],
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    print("── Forward pass with labels ──")
    output = model(input_ids, attention_mask, labels)
    print(f"  Loss:            {output['loss'].item():.4f}")
    print(f"  Logits shape:    {output['logits'].shape}")
    print(f"  Embeddings shape:{output['embeddings'].shape}")

    print("\n── Predictions ──")
    probs = torch.softmax(output["logits"], dim=1)
    preds = torch.argmax(probs, dim=1)
    label_names = config["labels"]["id2label"]
    for i, text in enumerate(texts):
        print(f"  Text:      {text[:50]}...")
        print(f"  Predicted: {label_names[preds[i].item()]} "
            f"(confidence: {probs[i].max().item():.2%})")
        print(f"  True:      {label_names[labels[i].item()]}")
        print()

    print("── Embedding extraction ──")
    embeddings = model.get_embeddings(input_ids, attention_mask)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embedding norm:   {embeddings.norm(dim=1).mean().item():.4f}")

    print("── Parameter count ──")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")

    print("\nHateBERT baseline smoke test passed!")