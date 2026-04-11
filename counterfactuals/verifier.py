# counterfactuals/verifier.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama
from dotenv import load_dotenv
from configs.config_loader import load_config
from counterfactuals.prompts import VERIFIER_PROMPT, VERIFIER_WITH_FEEDBACK_PROMPT

config = load_config()

VALID_LABELS = {"normal", "offensive", "hatespeech"}


def verify_label(text: str, previous_label: str = None) -> str:
    """
    Ask Llama-3.1-70B to classify a text.
    Returns: 'normal', 'offensive', or 'hatespeech'
    If previous_label is provided, uses the feedback prompt.
    """
    if previous_label:
        prompt = VERIFIER_WITH_FEEDBACK_PROMPT.format(
            text=text,
            previous_label=previous_label
        )
    else:
        prompt = VERIFIER_PROMPT.format(text=text)

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0,
            "num_predict": 10,   # same as max_tokens
        }
    )

    raw = response["message"]["content"].strip().lower()

    # Clean up any punctuation the model might add
    raw = raw.replace(".", "").replace(",", "").replace("'", "").strip()

    # Fuzzy match in case model says "hate speech" instead of "hatespeech"
    if raw in VALID_LABELS:
        return raw
    elif "hate" in raw:
        return "hatespeech"
    elif "offensive" in raw:
        return "offensive"
    else:
        return "normal"


def is_acceptable(text: str, required_label: str = "normal") -> tuple[bool, str]:
    """
    Verify that a counterfactual is actually non-harmful.
    Returns: (accepted: bool, predicted_label: str)
    """
    predicted = verify_label(text)
    accepted  = (predicted == required_label)
    return accepted, predicted


def verify_batch(
    texts: list[str],
    required_label: str = "normal",
    verbose: bool = True
) -> list[dict]:
    """
    Verify a list of counterfactuals.
    Returns list of dicts with text, predicted_label, accepted.
    """
    results = []
    for i, text in enumerate(texts):
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


if __name__ == "__main__":
    print("=== Verifier Smoke Test ===\n")

    test_cases = [
        ("I love spending time with people from different cultures.", "normal"),
        ("Those people are absolutely disgusting and should leave.", "hatespeech"),
        ("That was a pretty stupid thing to do honestly.",            "offensive"),
        ("People have different opinions on immigration policy.",     "normal"),
    ]

    print("── Single label verification ──")
    for text, expected in test_cases:
        predicted = verify_label(text)
        match = "✓" if predicted == expected else "✗"
        print(f"  [{match}] Expected: {expected:12s} | Got: {predicted:12s} | {text[:60]}...")

    print("\n── Batch verification ──")
    batch_texts = [t for t, _ in test_cases]
    verify_batch(batch_texts)

    print("\nverifier.py smoke test passed!")