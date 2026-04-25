# models/proposed_model.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from configs.config_loader import load_config
from training.contrastive_loss import CFContrastiveLoss, CombinedLoss

config = load_config()


class ProposedModel(nn.Module):
    """
    Model B — HateBERT with regular pairwise contrastive loss on CF pairs.
    
    Combines two objectives during training:
    1. Standard cross-entropy on original texts' labels
    2. Pairwise contrastive loss pushing (original, counterfactual) embeddings apart
    
    The contrastive component forces the model to learn semantic intent rather than
    surface-level patterns by directly penalizing similarity between harmful and non-harmful
    text embeddings.
    """

    def __init__(self, num_labels: int = 3):
        super().__init__()
        model_name = config["models"]["proposed"]["name"]

        print(f"Loading HateBERT for proposed model: {model_name}")
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
        Standard forward pass (used during evaluation or when no CF data available).
        
        Args:
            input_ids:      [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels:         [batch_size] optional

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

    def forward_pair(
        self,
        orig_input_ids:      torch.Tensor,
        orig_attention_mask: torch.Tensor,
        orig_labels:         torch.Tensor,
        cf_input_ids:        torch.Tensor,
        cf_attention_mask:   torch.Tensor,
        cf_labels:           torch.Tensor,
    ) -> dict:
        """
        Forward pass for CF pairs using supervised contrastive loss.
        Uses ALL in-batch relationships between examples of the same/different class.
        """
        from training.contrastive_loss import SupervisedContrastiveLoss
        import torch.nn as nn

        # Process originals
        orig_outputs = self.model(
            input_ids=orig_input_ids,
            attention_mask=orig_attention_mask,
            output_hidden_states=True,
        )
        orig_embeddings = orig_outputs.hidden_states[-1][:, 0, :]
        orig_logits     = orig_outputs.logits

        # Process counterfactuals
        cf_outputs = self.model(
            input_ids=cf_input_ids,
            attention_mask=cf_attention_mask,
            output_hidden_states=True,
        )
        cf_embeddings = cf_outputs.hidden_states[-1][:, 0, :]
        cf_logits     = cf_outputs.logits

        # CE loss on originals
        ce_loss = nn.CrossEntropyLoss()(orig_logits, orig_labels)

        # Supervised contrastive loss using all in-batch relationships
        sup_cont_loss = SupervisedContrastiveLoss()
        cont_loss = sup_cont_loss(
            orig_embeddings, cf_embeddings,
            orig_labels, cf_labels,
        )

        lambda_weight = config["models"]["proposed"]["contrastive_weight"]
        total_loss    = ce_loss + lambda_weight * cont_loss

        return {
            "loss": total_loss,
            "loss_breakdown": {
                "total":       total_loss.item(),
                "ce":          ce_loss.item(),
                "contrastive": cont_loss.item(),
                "lambda":      lambda_weight,
            },
            "orig_logits":     orig_logits,
            "orig_embeddings": orig_embeddings,
            "cf_logits":       cf_logits,
            "cf_embeddings":   cf_embeddings,
        }

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

    def get_predictions(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get class predictions and confidence scores.
        
        Returns:
            (predictions, confidences) — both [batch_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confidences = probs.max(dim=1).values
        return preds, confidences


def get_tokenizer():
    """Return the HateBERT tokenizer."""
    return AutoTokenizer.from_pretrained(config["models"]["proposed"]["name"])


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Proposed Model (Contrastive) Smoke Test ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model and tokenizer
    tokenizer = get_tokenizer()
    model     = ProposedModel(num_labels=3).to(device)

    # Sample texts
    orig_texts = [
        "I hate all immigrants they should be deported.",
        "Women are too stupid to be leaders.",
    ]
    orig_labels = torch.tensor([2, 1]).to(device)  # hate, offensive

    cf_texts = [
        "Different groups of people have diverse perspectives and contributions.",
        "Women have varied capabilities and skills like anyone else.",
    ]
    cf_labels = torch.tensor([0, 0]).to(device)  # normal, normal

    # Tokenize
    orig_encoding = tokenizer(
        orig_texts,
        padding=True,
        truncation=True,
        max_length=config["models"]["proposed"]["max_length"],
        return_tensors="pt",
    )
    cf_encoding = tokenizer(
        cf_texts,
        padding=True,
        truncation=True,
        max_length=config["models"]["proposed"]["max_length"],
        return_tensors="pt",
    )

    orig_input_ids      = orig_encoding["input_ids"].to(device)
    orig_attention_mask = orig_encoding["attention_mask"].to(device)
    cf_input_ids        = cf_encoding["input_ids"].to(device)
    cf_attention_mask   = cf_encoding["attention_mask"].to(device)

    print("── Standard forward pass (no CF) ──")
    output = model(orig_input_ids, orig_attention_mask, orig_labels)
    print(f"  Loss:            {output['loss'].item():.4f}")
    print(f"  Logits shape:    {output['logits'].shape}")
    print(f"  Embeddings shape:{output['embeddings'].shape}")

    print("\n── Forward pass with CF pairs (contrastive) ──")
    pair_output = model.forward_pair(
        orig_input_ids,
        orig_attention_mask,
        orig_labels,
        cf_input_ids,
        cf_attention_mask,
        cf_labels,
    )
    print(f"  Total loss:      {pair_output['loss'].item():.4f}")
    print(f"  Loss breakdown:")
    for key, val in pair_output["loss_breakdown"].items():
        if key != "lambda":
            print(f"    {key:15s}: {val:.4f}")
        else:
            print(f"    {key:15s}: {val:.2f}")

    print("\n── Embedding distances ──")
    orig_emb = pair_output["orig_embeddings"]
    cf_emb   = pair_output["cf_embeddings"]
    # Normalize and compute cosine similarity
    orig_norm = orig_emb / orig_emb.norm(dim=1, keepdim=True)
    cf_norm   = cf_emb / cf_emb.norm(dim=1, keepdim=True)
    similarity = (orig_norm * cf_norm).sum(dim=1)
    print(f"  Cosine similarity (original, CF): {similarity}")
    print(f"  Mean similarity: {similarity.mean().item():.4f}")
    print(f"  (Target: low/negative when trained properly)")

    print("\n── Predictions ──")
    preds, confs = model.get_predictions(orig_input_ids, orig_attention_mask)
    label_names = config["labels"]["id2label"]
    for i, text in enumerate(orig_texts):
        print(f"  Text:      {text[:50]}...")
        print(f"  Predicted: {label_names[preds[i].item()]} "
            f"(confidence: {confs[i].item():.2%})")
        print(f"  True:      {label_names[orig_labels[i].item()]}")
        print()

    print("── Parameter count ──")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")

    print("\nProposed model (contrastive) smoke test passed!")
