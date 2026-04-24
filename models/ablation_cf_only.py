# models/ablation_cf_only.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from configs.config_loader import load_config

config = load_config()


class AblationCFOnlyModel(nn.Module):
    """
    Model C (Ablation) — HateBERT trained on CF-augmented data WITHOUT contrastive loss.
    
    This ablation tests the effect of counterfactual data alone, independent of the
    contrastive loss component. During training, it:
    - Receives (original, counterfactual) pairs
    - Uses ONLY the original text for classification loss
    - Ignores the CF embeddings and applies standard cross-entropy
    
    If this model outperforms the baseline, it means CF data helps even without
    contrastive learning. If the proposed model significantly outperforms this,
    it indicates the contrastive component is critical.
    
    Research purpose: Isolate the contribution of contrastive learning vs. CF data.
    """

    def __init__(self, num_labels: int = 3):
        super().__init__()
        model_name = config["models"]["hatebert"]["name"]

        print(f"Loading HateBERT for ablation (CF-only): {model_name}")
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
        Standard forward pass.
        
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
        Forward pass during CF pair training (ablation version).
        
        KEY DIFFERENCE from proposed model: NO contrastive loss.
        We only use cross-entropy on the ORIGINAL texts.
        The CF texts are present in the batch but not used for loss computation.
        
        This tests: "Does having CF pairs help, even without contrastive learning?"
        
        Args:
            orig_input_ids:      [batch_size, seq_len]
            orig_attention_mask: [batch_size, seq_len]
            orig_labels:         [batch_size] — labels of original texts (1 or 2 for harmful)
            cf_input_ids:        [batch_size, seq_len] — CF texts (loaded for symmetry, not used)
            cf_attention_mask:   [batch_size, seq_len]
            cf_labels:           [batch_size] — labels of CFs (always 0, not used)

        Returns:
            dict with keys: loss, orig_logits, orig_embeddings, cf_logits, cf_embeddings
        """
        # Process originals
        orig_outputs = self.model(
            input_ids=orig_input_ids,
            attention_mask=orig_attention_mask,
            output_hidden_states=True,
        )
        orig_embeddings = orig_outputs.hidden_states[-1][:, 0, :]
        orig_logits     = orig_outputs.logits

        # Process counterfactuals (for logging/analysis only)
        cf_outputs = self.model(
            input_ids=cf_input_ids,
            attention_mask=cf_attention_mask,
            output_hidden_states=True,
        )
        cf_embeddings = cf_outputs.hidden_states[-1][:, 0, :]
        cf_logits     = cf_outputs.logits

        # ABLATION: Only compute CE loss on originals, no contrastive component
        ce_loss = nn.CrossEntropyLoss()(orig_logits, orig_labels)

        return {
            "loss": ce_loss,
            "loss_breakdown": {
                "total":        ce_loss.item(),
                "ce":           ce_loss.item(),
                "contrastive":  0.0,  # No contrastive loss in this ablation
                "lambda":       0.0,
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
    return AutoTokenizer.from_pretrained(config["models"]["hatebert"]["name"])


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Ablation Model (CF-Only, No Contrastive) Smoke Test ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model and tokenizer
    tokenizer = get_tokenizer()
    model     = AblationCFOnlyModel(num_labels=3).to(device)

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
        max_length=config["models"]["hatebert"]["max_length"],
        return_tensors="pt",
    )
    cf_encoding = tokenizer(
        cf_texts,
        padding=True,
        truncation=True,
        max_length=config["models"]["hatebert"]["max_length"],
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

    print("\n── Forward pass with CF pairs (NO contrastive loss — ablation) ──")
    pair_output = model.forward_pair(
        orig_input_ids,
        orig_attention_mask,
        orig_labels,
        cf_input_ids,
        cf_attention_mask,
        cf_labels,
    )
    print(f"  Loss (CE only):  {pair_output['loss'].item():.4f}")
    print(f"  Loss breakdown:")
    for key, val in pair_output["loss_breakdown"].items():
        if key != "lambda":
            print(f"    {key:15s}: {val:.4f}")
        else:
            print(f"    {key:15s}: {val:.2f}")
    print(f"  [Note: contrastive = 0.0 because this is the ablation]")

    print("\n── Original text predictions ──")
    preds, confs = model.get_predictions(orig_input_ids, orig_attention_mask)
    label_names = config["labels"]["id2label"]
    for i, text in enumerate(orig_texts):
        print(f"  Text:      {text[:50]}...")
        print(f"  Predicted: {label_names[preds[i].item()]} "
            f"(confidence: {confs[i].item():.2%})")
        print(f"  True:      {label_names[orig_labels[i].item()]}")
        print()

    print("── CF text predictions (for reference) ──")
    cf_preds, cf_confs = model.get_predictions(cf_input_ids, cf_attention_mask)
    for i, text in enumerate(cf_texts):
        print(f"  Text:      {text[:50]}...")
        print(f"  Predicted: {label_names[cf_preds[i].item()]} "
            f"(confidence: {cf_confs[i].item():.2%})")
        print(f"  True:      {label_names[cf_labels[i].item()]}")
        print()

    print("── Parameter count ──")
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")

    print("\nAblation model (CF-only) smoke test passed!")
