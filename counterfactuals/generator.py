# counterfactuals/generator.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from configs.config_loader import load_config
from counterfactuals.prompts import (
    ZERO_SHOT_CF_PROMPT,
    FEW_SHOT_CF_PROMPT,
    RETRY_CF_PROMPT,
    format_few_shot_examples,
)

config          = load_config()
GENERATOR_MODEL = "mistral"
OLLAMA_URL      = "http://localhost:11434/api/chat"


def _call_mistral(prompt: str) -> str:
    payload = {
        "model": GENERATOR_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": config["api"]["temperature"],
            "num_predict": config["api"]["max_tokens"],
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    text = response.json()["message"]["content"].strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[0] if lines else text


def generate_zero_shot(text: str, rationale_tokens: list = None) -> str:
    tokens_str = str(rationale_tokens) if rationale_tokens else "unknown — use your judgment"
    prompt = ZERO_SHOT_CF_PROMPT.format(
        text=text,
        rationale_tokens=tokens_str,
    )
    return _call_mistral(prompt)


def generate_few_shot(text: str, seed_examples: list,
                      rationale_tokens: list = None) -> str:
    tokens_str     = str(rationale_tokens) if rationale_tokens else "unknown — use your judgment"
    examples_str   = format_few_shot_examples(seed_examples)
    prompt = FEW_SHOT_CF_PROMPT.format(
        examples=examples_str,
        text=text,
        rationale_tokens=tokens_str,
    )
    return _call_mistral(prompt)


def generate_retry(original_text: str, previous_cf: str,
                   previous_label: str,
                   rationale_tokens: list = None) -> str:
    tokens_str = str(rationale_tokens) if rationale_tokens else "unknown — use your judgment"
    prompt = RETRY_CF_PROMPT.format(
        original_text=original_text,
        rationale_tokens=tokens_str,
        previous_cf=previous_cf,
        previous_label=previous_label,
    )
    return _call_mistral(prompt)


if __name__ == "__main__":
    print("=== Generator Smoke Test ===\n")
    test_text   = "Those immigrants are ruining our country."
    test_tokens = ["ruining"]

    print(f"Original:  {test_text}")
    print(f"Rationale: {test_tokens}")
    cf = generate_zero_shot(test_text, rationale_tokens=test_tokens)
    print(f"CF:        {cf}")