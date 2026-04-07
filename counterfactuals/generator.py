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


# ── Core API call ─────────────────────────────────────────────────────────────

def _call_mistral(prompt: str) -> str:
    """
    Send a prompt to local Mistral via Ollama.
    Returns the generated counterfactual text (stripped, first line only).
    """
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

    # Take only the first line — model sometimes adds explanation
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[0] if lines else text


# ── Public generation functions ───────────────────────────────────────────────

def generate_zero_shot(text: str) -> str:
    """Generate a counterfactual using zero-shot prompt."""
    prompt = ZERO_SHOT_CF_PROMPT.format(text=text)
    return _call_mistral(prompt)


def generate_few_shot(text: str, seed_examples: list[dict]) -> str:
    """Generate a counterfactual using few-shot prompt with seed examples."""
    examples_str = format_few_shot_examples(seed_examples)
    prompt = FEW_SHOT_CF_PROMPT.format(examples=examples_str, text=text)
    return _call_mistral(prompt)


def generate_retry(
    original_text:  str,
    previous_cf:    str,
    previous_label: str,
) -> str:
    """Generate a new counterfactual after a failed verification attempt."""
    prompt = RETRY_CF_PROMPT.format(
        original_text=original_text,
        previous_cf=previous_cf,
        previous_label=previous_label,
    )
    return _call_mistral(prompt)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Generator Smoke Test (Local Ollama) ===\n")
    print(f"Using model: {GENERATOR_MODEL} via Ollama\n")

    test_texts = [
        "Those immigrants are ruining our country and should all be deported.",
        "Women are too emotional to be good leaders.",
    ]

    print("── Zero-shot generation ──")
    for text in test_texts:
        print(f"\nOriginal:  {text}")
        cf = generate_zero_shot(text)
        print(f"CF:        {cf}")

    print("\n── Retry generation ──")
    original = "Those immigrants are ruining our country."
    bad_cf   = "Those immigrants are still causing problems."
    print(f"Original:   {original}")
    print(f"Bad CF:     {bad_cf}")
    retry_cf = generate_retry(original, bad_cf, "offensive")
    print(f"Retry CF:   {retry_cf}")

    print("\ngenerator.py smoke test done!")