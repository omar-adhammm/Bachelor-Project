# models/hatebert_baseline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from configs.config_loader import load_config

config = load_config()

# Class weights for weighted CE loss
# Computed as: total / (n_classes * class_count)
# normal=6494, offensive=4613, hate=5011, total=16118
CLASS_WEIGHTS = torch.tensor([0.827, 1.164, 1.072])


class HateBERTBaseline(nn.Module):
    """
    Model A — Phi-3.5-mini with LoRA fine-tuning on HateXplain.
    Uses generative classification: extracts last token hidden state
    and passes through a classification head.
    No counterfactual data, no contrastive loss.
    """

    def __init__(self, num_labels: int = 3):
        super().__init__()
        model_name = config["models"]["hatebert"]["name"]

        print(f"Loading HateBERT: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        # Apply LoRA
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            target_modules=config["lora"]["target_modules"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base_model, lora_config)

        # Classification head on top of last token hidden state
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels).float()
        self.num_labels  = num_labels

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         torch.Tensor = None,
        rationale_mask: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass. Extracts last token hidden state for classification.
        rationale_mask accepted but ignored in baseline.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Last token hidden state — for decoder models this is the prediction position
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]

        # Find last non-padding token for each example
        seq_lengths   = attention_mask.sum(dim=1) - 1  # [batch]
        last_hidden   = hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            seq_lengths,
        ]  # [batch, hidden]

        logits    = self.classifier(last_hidden.float())  # [batch, num_labels]
        embedding = last_hidden.float()

        result = {
            "logits":     logits,
            "embeddings": embedding,
        }

        if labels is not None:
            weights = CLASS_WEIGHTS.to(labels.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            result["loss"] = loss_fn(logits, labels)

        return result

    def get_embeddings(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states[-1]
        seq_lengths   = attention_mask.sum(dim=1) - 1
        return hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device),
            seq_lengths,
        ].float()


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["hatebert"]["name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder models need left padding
    return tokenizer


if __name__ == "__main__":
    print("=== HateBERT Baseline Smoke Test ===\n")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    tokenizer = get_tokenizer()
    model     = HateBERTBaseline(num_labels=3).to(device)

    texts  = [
        "I hate all immigrants they should be deported.",
        "That movie was absolutely terrible.",
        "I enjoyed spending time with my friends today.",
    ]
    labels = torch.tensor([2, 1, 0]).to(device)

    encoding = tokenizer(
        texts, padding=True, truncation=True,
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

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n── Parameter count ──")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")
    print("\nHateBERT baseline smoke test passed!")
