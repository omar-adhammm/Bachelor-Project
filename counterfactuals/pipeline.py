# counterfactuals/pipeline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import jsonlines
from tqdm import tqdm
from configs.config_loader import load_config
from counterfactuals.generator import generate_zero_shot, generate_few_shot, generate_retry
from counterfactuals.verifier import is_acceptable

config      = load_config()
MAX_RETRIES = config["counterfactual"]["max_retries"]
THRESHOLD   = config["counterfactual"]["acceptance_threshold"]
CF_OUTPUT   = config["paths"]["cf_pairs"]


# ── Single example CF generation with retry loop ──────────────────────────────

def generate_cf_for_example(
    example:       dict,
    strategy:      str = "zero_shot",
    seed_examples: list[dict] = None,
    verbose:       bool = False,
) -> dict:
    """
    Generate and verify a counterfactual for one harmful example.
    Returns a result dict with status, CF text, attempts, etc.
    """
    original_text  = example["text"]
    original_label = example["label"]

    result = {
        "id":            example["id"],
        "original_text": original_text,
        "original_label": original_label,
        "cf_text":       None,
        "cf_label":      "normal",
        "accepted":      False,
        "attempts":      0,
        "strategy_used": strategy,
    }

    previous_cf    = None
    previous_label = None

    for attempt in range(1, MAX_RETRIES + 1):
        result["attempts"] = attempt

        try:
            if attempt == 1:
                if strategy == "few_shot" and seed_examples:
                    cf_text = generate_few_shot(original_text, seed_examples)
                else:
                    cf_text = generate_zero_shot(original_text)
            else:
                cf_text = generate_retry(original_text, previous_cf, previous_label)

            accepted, predicted = is_acceptable(cf_text, required_label="normal")

            if verbose:
                status = "✓ ACCEPTED" if accepted else f"✗ rejected ({predicted})"
                print(f"    Attempt {attempt}: {status} | {cf_text[:70]}...")

            if accepted:
                result["cf_text"]  = cf_text
                result["accepted"] = True
                return result

            previous_cf    = cf_text
            previous_label = predicted
            time.sleep(0.1)

        except Exception as e:
            if verbose:
                print(f"    Attempt {attempt} ERROR: {e}")
            time.sleep(0.5)
            continue

    result["cf_text"] = previous_cf
    return result


# ── Load already processed IDs from partial output ───────────────────────────

def load_already_processed(output_path: str) -> tuple[set, list, int]:
    """
    Read an existing partial output file.
    Returns:
        already_done_ids  — set of IDs already saved to disk
        accepted_so_far   — list of accepted result dicts for seed pool reuse
        accepted_count    — number of accepted so far
    """
    already_done_ids = set()
    accepted_so_far  = []
    accepted_count   = 0

    if not os.path.exists(output_path):
        return already_done_ids, accepted_so_far, accepted_count

    with jsonlines.open(output_path) as reader:
        for obj in reader:
            ex_id = obj["original"]["id"]
            already_done_ids.add(ex_id)
            accepted_so_far.append({
                "original_text": obj["original"]["text"],
                "cf_text":       obj["counterfactual"]["text"],
                "accepted":      True,
            })
            accepted_count += 1

    print(f"Resuming — found {accepted_count} already accepted CFs on disk.")
    print(f"Skipping {len(already_done_ids)} already processed examples.\n")
    return already_done_ids, accepted_so_far, accepted_count


# ── Write one accepted result to disk immediately ─────────────────────────────

def save_result(writer, result: dict) -> None:
    """Write one accepted CF pair to the open jsonlines writer."""
    writer.write({
        "original": {
            "id":       result["id"],
            "text":     result["original_text"],
            "label":    result["original_label"],
            "label_id": config["labels"]["label2id"][result["original_label"]],
        },
        "counterfactual": {
            "text":     result["cf_text"],
            "label":    "normal",
            "label_id": 0,
        },
        "attempts":      result["attempts"],
        "strategy_used": result["strategy_used"],
    })


# ── Full dataset pipeline ─────────────────────────────────────────────────────

def run_pipeline(
    split:        str  = "train",
    max_examples: int  = None,
    verbose:      bool = False,
) -> dict:
    """
    Run CF generation on all harmful examples in a split.
    Saves results incrementally to outputs/cf_pairs/{split}_cf_pairs.jsonl
    Resumes automatically if a partial file already exists.
    Returns a statistics dict.
    """
    # ── Setup output path ─────────────────────────────────────────────────────
    os.makedirs(CF_OUTPUT, exist_ok=True)
    output_path = os.path.join(CF_OUTPUT, f"{split}_cf_pairs.jsonl")

    # ── Load split ────────────────────────────────────────────────────────────
    with open(os.path.join(config["paths"]["raw_data"], f"{split}.json")) as f:
        examples = json.load(f)

    harmful = [ex for ex in examples if ex["label"] in config["labels"]["harmful"]]

    if max_examples:
        harmful = harmful[:max_examples]

    # ── Resume check ──────────────────────────────────────────────────────────
    already_done_ids, accepted_so_far, accepted_count = load_already_processed(output_path)
    remaining = [ex for ex in harmful if ex["id"] not in already_done_ids]

    print(f"Running CF pipeline on '{split}' split")
    print(f"  Total harmful:   {len(harmful)}")
    print(f"  Already done:    {len(already_done_ids)}")
    print(f"  To process now:  {len(remaining)}")
    print(f"  Max retries:     {MAX_RETRIES}")
    print(f"  Strategy:        {config['counterfactual']['strategy']}\n")

    if not remaining:
        print("All examples already processed. Nothing to do.")
        stats = _compute_stats(split, harmful, accepted_count, output_path)
        _print_stats(stats)
        return stats

    # ── Phase 1: zero-shot run ────────────────────────────────────────────────
    # Open file in APPEND mode — preserves existing accepted pairs
    all_results = []
    zero_shot_accepted = 0

    with jsonlines.open(output_path, mode="a") as writer:
        for example in tqdm(remaining, desc="Generating CFs (zero-shot)"):
            result = generate_cf_for_example(
                example,
                strategy="zero_shot",
                verbose=verbose,
            )
            all_results.append(result)

            if result["accepted"]:
                accepted_count += 1
                zero_shot_accepted += 1
                save_result(writer, result)

            # Brief pause to stay within API rate limits
            time.sleep(0.05)

    zero_shot_rate = zero_shot_accepted / len(remaining) if remaining else 1.0
    print(f"\nZero-shot acceptance rate: {zero_shot_accepted}/{len(remaining)} "
          f"({zero_shot_rate * 100:.1f}%)")

    # ── Phase 2: few-shot fallback if rate is below threshold ─────────────────
    if zero_shot_rate < THRESHOLD:
        print(f"\nAcceptance rate below {THRESHOLD*100:.0f}% threshold.")
        print("Activating few-shot fallback for rejected examples...")

        # Build seed pool from all accepted so far (existing + new)
        seed_pool = [
            {
                "original":       r["original_text"],
                "counterfactual": r["cf_text"],
            }
            for r in all_results if r["accepted"]
        ]

        # Also include any previously saved accepted pairs
        seed_pool += [
            {
                "original":       r["original_text"],
                "counterfactual": r["cf_text"],
            }
            for r in accepted_so_far
        ]

        seed_pool = seed_pool[:5]  # cap at 5 examples

        if len(seed_pool) < 3:
            print(f"Warning: only {len(seed_pool)} seed examples available.")
            print("Proceeding with available seeds.")

        rejected = [r for r in all_results if not r["accepted"]]
        print(f"Re-running {len(rejected)} rejected examples with few-shot...\n")

        few_shot_accepted = 0

        with jsonlines.open(output_path, mode="a") as writer:
            for result in tqdm(rejected, desc="Generating CFs (few-shot)"):
                example = {
                    "id":       result["id"],
                    "text":     result["original_text"],
                    "label":    result["original_label"],
                    "label_id": config["labels"]["label2id"][result["original_label"]],
                }
                new_result = generate_cf_for_example(
                    example,
                    strategy="few_shot",
                    seed_examples=seed_pool,
                    verbose=verbose,
                )

                if new_result["accepted"]:
                    accepted_count += 1
                    few_shot_accepted += 1
                    save_result(writer, new_result)

                time.sleep(0.05)

        print(f"Few-shot added {few_shot_accepted} more accepted CFs.")

    # ── Final statistics ──────────────────────────────────────────────────────
    stats = _compute_stats(split, harmful, accepted_count, output_path)
    _print_stats(stats)
    return stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_stats(split, harmful, accepted_count, output_path) -> dict:
    return {
        "split":           split,
        "total_harmful":   len(harmful),
        "accepted":        accepted_count,
        "rejected":        len(harmful) - accepted_count,
        "acceptance_rate": accepted_count / len(harmful) if harmful else 0,
        "output_path":     output_path,
    }


def _print_stats(stats: dict) -> None:
    print(f"\n── Final Statistics ──")
    print(f"  Total harmful:    {stats['total_harmful']}")
    print(f"  Accepted CFs:     {stats['accepted']}")
    print(f"  Rejected:         {stats['rejected']}")
    print(f"  Acceptance rate:  {stats['acceptance_rate']*100:.1f}%")
    print(f"  Saved to:         {stats['output_path']}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max",     type=int, default=None,
                        help="Limit number of examples (for testing)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"=== CF Pipeline: {args.split} split ===\n")
    stats = run_pipeline(
        split=args.split,
        max_examples=args.max,
        verbose=args.verbose,
    )