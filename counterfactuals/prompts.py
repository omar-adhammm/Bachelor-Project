# counterfactuals/prompts.py

ZERO_SHOT_CF_PROMPT = """You are a text editor. Your task is to change ONLY the harmful words in the text below to make it non-harmful.

Text: {text}
Harmful words to change: {rationale_tokens}

Rules:
1. ONLY change the words listed in "Harmful words to change" — do not touch anything else
2. Replace each harmful word with a neutral alternative that fits naturally
3. Keep every other word exactly the same
4. Keep the same sentence length and structure
5. Output ONLY the modified text — no explanation, no quotes, no preamble

Modified text:"""


FEW_SHOT_CF_PROMPT = """You are a text editor. Your task is to change ONLY the harmful words in text to make it non-harmful.

Study these examples — notice how only the harmful words are changed:

{examples}

Now apply the same approach:

Text: {text}
Harmful words to change: {rationale_tokens}

Rules:
1. ONLY change the words listed in "Harmful words to change" — do not touch anything else
2. Replace each harmful word with a neutral alternative that fits naturally
3. Keep every other word exactly the same
4. Output ONLY the modified text — no explanation, no quotes, no preamble

Modified text:"""


VERIFIER_PROMPT = """You are a hate speech classifier. Classify the following text into exactly one of these categories:
- normal
- offensive
- hatespeech

Text: {text}

Respond with ONLY one word — either: normal, offensive, or hatespeech"""


VERIFIER_WITH_FEEDBACK_PROMPT = """You are a hate speech classifier. Classify the following text into exactly one of these categories:
- normal
- offensive
- hatespeech

Text: {text}

Previous classification attempt said: {previous_label}
Reconsider carefully.

Respond with ONLY one word — either: normal, offensive, or hatespeech"""


RETRY_CF_PROMPT = """You are a text editor. Your previous rewrite was still classified as harmful.

Original text: {original_text}
Harmful words to change: {rationale_tokens}
Your previous attempt: {previous_cf}
Problem: your rewrite was classified as {previous_label} — not as normal.

Try again. You MUST only change the words listed in "Harmful words to change".
Keep everything else exactly the same.

Output ONLY the modified text — no explanation, no quotes, no preamble

Modified text:"""


def format_few_shot_examples(seed_examples: list) -> str:
    lines = []
    for i, ex in enumerate(seed_examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Original:       {ex['original']}")
        lines.append(f"  Harmful words:  {ex.get('rationale_tokens', 'N/A')}")
        lines.append(f"  Modified:       {ex['counterfactual']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== ZERO SHOT PROMPT ===")
    print(ZERO_SHOT_CF_PROMPT.format(
        text="those immigrants are ruining our country",
        rationale_tokens="['ruining']"
    ))

    print("\n=== RETRY PROMPT ===")
    print(RETRY_CF_PROMPT.format(
        original_text="those immigrants are ruining our country",
        rationale_tokens="['ruining']",
        previous_cf="those immigrants are destroying our country",
        previous_label="offensive"
    ))