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
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            attn_implementation="eager",
        )

        # Apply LoRA
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            target_modules=config["lora"]["target_modules"],
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"  LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        self.num_labels = num_labels

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         torch.Tensor = None,
        rationale_mask: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass with optional rationale supervision.
        """
        from training.contrastive_loss import RationaleSupervisionLoss

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,   # NEW — get attention weights
            output_hidden_states=True,  # NEW — get hidden states for rationale loss
        )

        cls_embedding = output.hidden_states[-1][:, 0, :]

        result = {
            "logits": output.logits,
            "embeddings": cls_embedding,
        }

        if labels is not None:
            weights = torch.tensor([0.827, 1.164, 1.072], device=labels.device)
            result["loss"] = nn.CrossEntropyLoss(weight=weights)(output.logits, labels)
        else:
            result["loss"] = None

        # Add rationale supervision if mask provided
        if rationale_mask is not None and labels is not None:
            rat_weight = config["models"]["proposed"].get("rationale_weight", 0.2)

            # Extract CLS token attention from last layer
            # Shape: [batch, n_heads, seq_len, seq_len]
            last_layer_attn = output.attentions[-1]
            # Average across heads, take CLS token (position 0) attention
            cls_attn = last_layer_attn[:, :, 0, :].mean(dim=1)  # [batch, seq_len]

            rat_loss_fn = RationaleSupervisionLoss()
            rat_loss    = rat_loss_fn(cls_attn, rationale_mask, labels)

            result["loss"]         = result["loss"] + rat_weight * rat_loss
            result["rationale_loss"] = rat_loss.item()

        return result

    def forward_pair(
        self,
        orig_input_ids:      torch.Tensor,
        orig_attention_mask: torch.Tensor,
        orig_labels:         torch.Tensor,
        cf_input_ids:        torch.Tensor,
        cf_attention_mask:   torch.Tensor,
        cf_labels:           torch.Tensor,
        orig_rationale_mask: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass with rationale-guided contrastive loss.
        If orig_rationale_mask is provided, uses rationale token embeddings.
        Falls back to CLS-based pairwise loss if not provided.
        """
        from training.contrastive_loss import (
            RationaleGuidedContrastiveLoss,
            CFContrastiveLoss,
        )

        # Process originals — need full hidden states for rationale extraction
        orig_outputs = self.model(
            input_ids=orig_input_ids,
            attention_mask=orig_attention_mask,
            output_hidden_states=True,
        )
        # LoRA wraps the model — access hidden states safely
        if hasattr(orig_outputs, 'hidden_states') and orig_outputs.hidden_states is not None:
            orig_hidden = orig_outputs.hidden_states[-1]
        else:
            # Fallback: use a separate forward pass to get hidden states
            orig_hidden = self.model.base_model(
                input_ids=orig_input_ids,
                attention_mask=orig_attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1]
        orig_emb    = orig_hidden[:, 0, :]
        orig_logits = orig_outputs.logits

        # Process counterfactuals
        cf_outputs = self.model(
            input_ids=cf_input_ids,
            attention_mask=cf_attention_mask,
            output_hidden_states=True,
        )
        if hasattr(cf_outputs, 'hidden_states') and cf_outputs.hidden_states is not None:
            cf_hidden = cf_outputs.hidden_states[-1]
        else:
            cf_hidden = self.model.base_model(
                input_ids=cf_input_ids,
                attention_mask=cf_attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1]
        cf_emb    = cf_hidden[:, 0, :]
        cf_logits = cf_outputs.logits

        # CE loss on originals
        ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.827, 1.164, 1.072], device=orig_labels.device))(orig_logits, orig_labels)

        # Choose contrastive loss
        if orig_rationale_mask is not None:
            # Rationale-guided: use rationale token embeddings
            cont_fn   = RationaleGuidedContrastiveLoss()
            cont_loss = cont_fn(
                orig_hidden, cf_hidden,
                orig_rationale_mask,
                orig_attention_mask,
                cf_attention_mask,
            )
            loss_type = "rationale_guided"
        else:
            # Fallback: pairwise CLS contrastive
            cont_fn   = CFContrastiveLoss()
            cont_loss = cont_fn(orig_emb, cf_emb)
            loss_type = "pairwise_cls"

        lambda_weight = config["models"]["proposed"]["contrastive_weight"]
        total_loss    = ce_loss + lambda_weight * cont_loss

        return {
            "loss": total_loss,
            "loss_breakdown": {
                "total":       total_loss.item(),
                "ce":          ce_loss.item(),
                "contrastive": cont_loss.item(),
                "lambda":      lambda_weight,
                "loss_type":   loss_type,
            },
            "orig_logits":     orig_logits,
            "orig_embeddings": orig_emb,
            "cf_logits":       cf_logits,
            "cf_embeddings":   cf_emb,
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
        if isinstance(val, str):
            print(f"    {key:15s}: {val}")
        elif key == "lambda":
            print(f"    {key:15s}: {val:.2f}")
        else:
            print(f"    {key:15s}: {val:.4f}")

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
