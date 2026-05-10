# models/ablation_cf_only.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from configs.config_loader import load_config

config = load_config()

CLASS_WEIGHTS = torch.tensor([0.827, 1.164, 1.072])


class AblationCFOnlyModel(nn.Module):
    """
    Model C — Phi-3.5-mini with LoRA + CF data but NO contrastive loss.
    Isolates contribution of CF data vs contrastive objective.
    """

    def __init__(self, num_labels: int = 3):
        super().__init__()
        model_name = config["models"]["hatebert"]["name"]

        print(f"Loading HateBERT for ablation (CF-only): {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            attn_implementation="eager",
        )

        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            lora_dropout=config["lora"]["lora_dropout"],
            target_modules=config["lora"]["target_modules"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base_model, lora_config)

        hidden_size      = self.model.config.hidden_size
        self.classifier  = nn.Linear(hidden_size, num_labels)
        self.num_labels  = num_labels

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _get_last_hidden(self, input_ids, attention_mask):
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

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         torch.Tensor = None,
        rationale_mask: torch.Tensor = None,
    ) -> dict:
        _ = rationale_mask
        embedding = self._get_last_hidden(input_ids, attention_mask)
        logits    = self.classifier(embedding)

        result = {"logits": logits, "embeddings": embedding}

        if labels is not None:
            weights = CLASS_WEIGHTS.to(labels.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            result["loss"] = loss_fn(logits, labels)

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
        _ = cf_labels, orig_rationale_mask
        orig_emb    = self._get_last_hidden(orig_input_ids, orig_attention_mask)
        orig_logits = self.classifier(orig_emb)
        cf_emb      = self._get_last_hidden(cf_input_ids, cf_attention_mask)
        cf_logits   = self.classifier(cf_emb)

        weights = CLASS_WEIGHTS.to(orig_labels.device)
        ce_loss = nn.CrossEntropyLoss(weight=weights)(orig_logits, orig_labels)

        return {
            "loss": ce_loss,
            "loss_breakdown": {
                "total":       ce_loss.item(),
                "ce":          ce_loss.item(),
                "contrastive": 0.0,
                "lambda":      0.0,
            },
            "orig_logits":     orig_logits,
            "orig_embeddings": orig_emb,
            "cf_logits":       cf_logits,
            "cf_embeddings":   cf_emb,
        }

    def get_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            return self._get_last_hidden(input_ids, attention_mask)


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["hatebert"]["name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


if __name__ == "__main__":
    print("=== Ablation Model Smoke Test ===\n")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    model     = AblationCFOnlyModel(num_labels=3).to(device)

    texts  = ["I hate immigrants.", "Women can not lead."]
    labels = torch.tensor([2, 1]).to(device)
    cf_texts  = ["Immigration is complex.", "Women have diverse strengths."]
    cf_labels = torch.tensor([0, 0]).to(device)

    enc    = tokenizer(texts, padding=True, truncation=True,
                       max_length=64, return_tensors="pt")
    cf_enc = tokenizer(cf_texts, padding=True, truncation=True,
                       max_length=64, return_tensors="pt")

    out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), labels)
    print(f"Forward loss: {out['loss'].item():.4f}")

    pair_out = model.forward_pair(
        enc["input_ids"].to(device), enc["attention_mask"].to(device), labels,
        cf_enc["input_ids"].to(device), cf_enc["attention_mask"].to(device), cf_labels,
    )
    print(f"Pair loss: {pair_out['loss'].item():.4f}")
    print("Ablation smoke test passed!")
