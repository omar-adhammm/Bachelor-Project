# counterfactuals/verifier_colab.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.config_loader import load_config
from counterfactuals.prompts import (
    VERIFIER_PROMPT,
    VERIFIER_WITH_FEEDBACK_PROMPT,
)

config         = load_config()
VERIFIER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VALID_LABELS   = {"normal", "offensive", "hatespeech"}

print(f"Loading verifier: {VERIFIER_MODEL}...")
_tokenizer = AutoTokenizer.from_pretrained(VERIFIER_MODEL)
_model     = AutoModelForCausalLM.from_pretrained(
    VERIFIER_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
_model.eval()
print("Verifier loaded!")


def verify_label(text: str, previous_label: str = None) -> str:
    if previous_label:
        prompt = VERIFIER_WITH_FEEDBACK_PROMPT.format(
            text=text,
            previous_label=previous_label,
        )
    else:
        prompt = VERIFIER_PROMPT.format(text=text)

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=10,       # only needs one word
            temperature=0.01,        # near-deterministic
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
    raw = raw.replace(".", "").replace(",", "").strip()

    if raw in VALID_LABELS:
        return raw
    elif "hate" in raw:
        return "hatespeech"
    elif "offensive" in raw:
        return "offensive"
    else:
        return "normal"


def is_acceptable(text: str, required_label: str = "normal") -> tuple[bool, str]:
    predicted = verify_label(text)
    return (predicted == required_label), predicted


def verify_batch(
    texts: list[str],
    required_label: str = "normal",
    verbose: bool = True,
) -> list[dict]:
    results = []
    for text in texts:
        accepted, predicted = is_acceptable(text, required_label)
        results.append({
            "text":            text,
            "predicted_label": predicted,
            "accepted":        accepted,
        })
        if verbose:
            status = "✓" if accepted else "✗"
            print(f"  [{status}] ({predicted:12s}) {text[:70]}...")

    accepted_count = sum(r["accepted"] for r in results)
    if verbose:
        print(f"\nAcceptance rate: {accepted_count}/{len(texts)} "
              f"({accepted_count/len(texts)*100:.1f}%)")
    return results