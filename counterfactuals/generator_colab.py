# counterfactuals/generator_colab.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.config_loader import load_config
from counterfactuals.prompts import (
    ZERO_SHOT_CF_PROMPT,
    FEW_SHOT_CF_PROMPT,
    RETRY_CF_PROMPT,
    format_few_shot_examples,
)

config          = load_config()
GENERATOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Loading generator: {GENERATOR_MODEL}...")
_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
_model     = AutoModelForCausalLM.from_pretrained(
    GENERATOR_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
_model.eval()
print("Generator loaded!")


def _call_mistral(prompt: str) -> str:
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=config["api"]["temperature"],
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[0] if lines else text


def generate_zero_shot(text: str) -> str:
    prompt = ZERO_SHOT_CF_PROMPT.format(text=text)
    return _call_mistral(prompt)


def generate_few_shot(text: str, seed_examples: list[dict]) -> str:
    examples_str = format_few_shot_examples(seed_examples)
    prompt = FEW_SHOT_CF_PROMPT.format(examples=examples_str, text=text)
    return _call_mistral(prompt)


def generate_retry(
    original_text:  str,
    previous_cf:    str,
    previous_label: str,
) -> str:
    prompt = RETRY_CF_PROMPT.format(
        original_text=original_text,
        previous_cf=previous_cf,
        previous_label=previous_label,
    )
    return _call_mistral(prompt)