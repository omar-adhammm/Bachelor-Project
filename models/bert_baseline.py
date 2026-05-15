# models/bert_baseline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from configs.config_loader import load_config

config = load_config()

CLASS_WEIGHTS = torch.tensor([0.827, 1.164, 1.072])


class BERTBaseline(nn.Module):
    """
    Standard BERT-base-uncased fine-tuned on HateXplain.
    Matches the setup of Mathew et al. 2021 as closely as possible.
    Uses weighted CE loss for fair comparison with other models.
    No LoRA — full fine-tuning to match the 2021 paper.
    """

    def __init__(self, num_labels: int = 3):
        super().__init__()
        model_name = "bert-base-uncased"
        print(f"Loading BERT Baseline: {model_name}")

        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout    = nn.Dropout(0.1)
        self._model_name_key = 'bert'
        self.num_labels = num_labels

        total = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total:,} (full fine-tuning)")

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         torch.Tensor = None,
        **kwargs,
    ) -> dict:
        output    = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use CLS token embedding
        cls_emb   = output.last_hidden_state[:, 0, :]
        cls_emb   = self.dropout(cls_emb)
        logits    = self.classifier(cls_emb)

        result = {"logits": logits, "embeddings": cls_emb}

        if labels is not None:
            weights = CLASS_WEIGHTS.to(labels.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            result["loss"] = loss_fn(logits, labels)

        return result


def get_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")
