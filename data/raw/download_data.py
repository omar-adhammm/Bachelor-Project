import json
import random
import urllib.request
import os
from collections import Counter


def build_examples(raw: dict) -> dict:
    """Build processed example dict from raw HateXplain data. Returns id->example map."""
    label_map = {"normal": 0, "offensive": 1, "hatespeech": 2}
    examples  = {}

    for post_id, item in raw.items():
        labels   = [a["label"] for a in item["annotators"]]
        majority = Counter(labels).most_common(1)[0][0]

        tokens     = item["post_tokens"]
        rationales = item.get("rationales", [])

        if rationales:
            n_tokens      = len(tokens)
            avg_rationale = [0.0] * n_tokens
            for rat in rationales:
                for j, val in enumerate(rat[:n_tokens]):
                    avg_rationale[j] += val
            avg_rationale = [v / len(rationales) for v in avg_rationale]
            rationale_mask = [1 if v >= 0.5 else 0 for v in avg_rationale]
        else:
            rationale_mask = [0] * len(tokens)

        examples[post_id] = {
            "id":             post_id,
            "text":           " ".join(tokens),
            "label":          majority,
            "label_id":       label_map.get(majority, -1),
            "annotators":     item["annotators"],
            "rationales":     item.get("rationales", []),
            "rationale_mask": rationale_mask,
        }

    return examples


def get_official_splits():
    """
    Download the official HateXplain train/val/test split IDs
    directly from the HateXplain GitHub repository.
    """
    url = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"
    print(f"Downloading official split IDs from HateXplain GitHub...")
    with urllib.request.urlopen(url) as response:
        split_data = json.loads(response.read().decode())

    train_ids = set(split_data["train"])
    val_ids   = set(split_data["val"])
    test_ids  = set(split_data["test"])

    print(f"Official split sizes:")
    print(f"  Train: {len(train_ids)}")
    print(f"  Val:   {len(val_ids)}")
    print(f"  Test:  {len(test_ids)}")

    return train_ids, val_ids, test_ids


def reorganize_cf_pairs(old_cf_dir, new_train_ids, new_val_ids, new_test_ids, out_dir):
    """
    Reorganize existing CF pairs according to new split assignments.
    Reads all CF pairs from all split files and redistributes them.
    """
    all_pairs = []

    for split in ["train", "validation", "test"]:
        cf_path = os.path.join(old_cf_dir, f"{split}_cf_pairs.jsonl")
        if not os.path.exists(cf_path):
            print(f"  Warning: {cf_path} not found, skipping")
            continue
        with open(cf_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_pairs.append(json.loads(line))

    print(f"\nTotal CF pairs loaded: {len(all_pairs)}")

    train_pairs, val_pairs, test_pairs, unmatched = [], [], [], []

    for pair in all_pairs:
        orig_id = pair["original"]["id"]
        if orig_id in new_train_ids:
            train_pairs.append(pair)
        elif orig_id in new_val_ids:
            val_pairs.append(pair)
        elif orig_id in new_test_ids:
            test_pairs.append(pair)
        else:
            unmatched.append(orig_id)

    print(f"Redistributed CF pairs:")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Val:   {len(val_pairs)}")
    print(f"  Test:  {len(test_pairs)}")
    if unmatched:
        print(f"  Unmatched (skipped): {len(unmatched)}")

    os.makedirs(out_dir, exist_ok=True)
    for split_name, pairs in [("train", train_pairs),
                               ("validation", val_pairs),
                               ("test", test_pairs)]:
        out_path = os.path.join(out_dir, f"{split_name}_cf_pairs.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        print(f"  Saved: {out_path}")


def main(use_official: bool = False):
    with open("data/raw/dataset.json") as f:
        raw = json.load(f)

    # Build all examples
    id_to_example = build_examples(raw)
    all_data      = list(id_to_example.values())

    if use_official:
        # Use official HateXplain predefined splits
        train_ids, val_ids, test_ids = get_official_splits()

        train = [id_to_example[i] for i in train_ids if i in id_to_example]
        val   = [id_to_example[i] for i in val_ids   if i in id_to_example]
        test  = [id_to_example[i] for i in test_ids  if i in id_to_example]

        print(f"\nUsing official HateXplain splits")

        # Save to official subfolder
        save_dir = "data/raw/official"
        os.makedirs(save_dir, exist_ok=True)

        for name, data in [("train", train), ("validation", val), ("test", test)]:
            with open(f"{save_dir}/{name}.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data)} examples → {save_dir}/{name}.json")

        # Reorganize existing CF pairs to match new splits
        reorganize_cf_pairs(
            old_cf_dir="outputs/cf_pairs",
            new_train_ids=train_ids,
            new_val_ids=val_ids,
            new_test_ids=test_ids,
            out_dir="outputs/cf_pairs_official",
        )

        # Check rationale coverage
        has_rationale = sum(1 for ex in train if any(ex["rationale_mask"]))
        print(f"\nTrain examples with rationales: {has_rationale}/{len(train)} "
              f"({has_rationale/len(train)*100:.1f}%)")

    else:
        # Original random 80/10/10 split
        random.seed(42)
        random.shuffle(all_data)

        n       = len(all_data)
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)

        train = all_data[:n_train]
        val   = all_data[n_train:n_train + n_val]
        test  = all_data[n_train + n_val:]

        for name, data in [("train", train), ("validation", val), ("test", test)]:
            with open(f"data/raw/{name}.json", "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data)} examples → data/raw/{name}.json")

        has_rationale = sum(1 for ex in train if any(ex["rationale_mask"]))
        print(f"\nTrain examples with rationales: {has_rationale}/{len(train)} "
              f"({has_rationale/len(train)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official_split",
        action="store_true",
        help="Use official HateXplain splits instead of random 80/10/10"
    )
    args = parser.parse_args()
    main(use_official=args.official_split)