# counterfactuals/prompts.py

ZERO_SHOT_CF_PROMPT = """You are a text editor. Your task is to rewrite a harmful text into a non-harmful version, output only the rewritten sentence, nothing else, and follow these rules:

Rules:
1. Keep the SAME topic and context
2. Remove or neutralize the harmful intent
3. Keep similar sentence length and structure
4. Output ONLY the rewritten text — no explanation, no quotes, no preamble

Original text: {text}

Rewritten version:"""


FEW_SHOT_CF_PROMPT = """You are a text editor. Your task is to rewrite harmful text into non-harmful versions.
Study these examples first:

{examples}

Now rewrite the following text using the same approach:

Rules:
1. Keep the SAME topic and context
2. Remove or neutralize the harmful intent
3. Keep similar sentence length and structure
4. Output ONLY the rewritten text — no explanation, no quotes, no preamble

Original text: {text}

Rewritten version:"""


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


RETRY_CF_PROMPT = """You are a text editor. Your previous rewrite was not good enough.

Original harmful text: {original_text}
Your previous rewrite: {previous_cf}
Problem: your rewrite was still classified as {previous_label}, not as normal/non-harmful.

Try again. Make more significant changes to remove the harmful content, output only the rewritten sentence, and follow these rules:

Rules:
1. Keep the SAME topic and context
2. Completely remove or neutralize the harmful intent
3. Keep similar sentence length and structure  
4. Output ONLY the rewritten text — no explanation, no quotes, no preamble

Rewritten version:"""


def format_few_shot_examples(seed_examples: list[dict]) -> str:
    """
    Format seed examples for the few-shot prompt.
    Each seed must have: original (str), counterfactual (str)
    """
    lines = []
    for i, ex in enumerate(seed_examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Original:  {ex['original']}")
        lines.append(f"  Rewritten: {ex['counterfactual']}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick visual check
    print("=== ZERO SHOT PROMPT ===")
    print(ZERO_SHOT_CF_PROMPT.format(text="Sample hate speech text here"))

    print("\n=== VERIFIER PROMPT ===")
    print(VERIFIER_PROMPT.format(text="Sample text to classify"))

    print("\n=== RETRY PROMPT ===")
    print(RETRY_CF_PROMPT.format(
        original_text="Original bad text",
        previous_cf="Still bad rewrite",
        previous_label="offensive"
    ))

    print("\n=== FEW SHOT EXAMPLES FORMAT ===")
    seeds = [
        {"original": "I hate all X people",     "counterfactual": "I disagree with some people's views"},
        {"original": "All Y should be removed",  "counterfactual": "Society should address certain challenges"},
    ]
    print(FEW_SHOT_CF_PROMPT.format(
        examples=format_few_shot_examples(seeds),
        text="New text to rewrite"
    ))

    print("\nprompts.py loaded successfully!")