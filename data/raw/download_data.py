import json
import random
from collections import Counter

with open("data/raw/dataset.json") as f:
    raw = json.load(f)

label_map = {"normal": 0, "offensive": 1, "hatespeech": 2}

all_data = []
for post_id, item in raw.items():
    labels = [a["label"] for a in item["annotators"]]
    majority = Counter(labels).most_common(1)[0][0]

    # Build majority-vote rationale mask
    # Average annotator rationales and threshold at 0.5
    tokens     = item["post_tokens"]
    rationales = item.get("rationales", [])

    if rationales:
        n_tokens = len(tokens)
        # Average across annotators
        avg_rationale = [0.0] * n_tokens
        for rat in rationales:
            for j, val in enumerate(rat[:n_tokens]):
                avg_rationale[j] += val
        avg_rationale = [v / len(rationales) for v in avg_rationale]
        # Threshold at 0.5 — majority of annotators must agree
        rationale_mask = [1 if v >= 0.5 else 0 for v in avg_rationale]
    else:
        rationale_mask = [0] * len(tokens)

    entry = {
        "id":            post_id,
        "text":          " ".join(tokens),
        "label":         majority,
        "label_id":      label_map.get(majority, -1),
        "annotators":    item["annotators"],
        "rationales":    item.get("rationales", []),
        "rationale_mask": rationale_mask,  # NEW
    }
    all_data.append(entry)

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

# Check rationale coverage
has_rationale = sum(1 for ex in train if any(ex["rationale_mask"]))
print(f"\nTrain examples with rationales: {has_rationale}/{len(train)} "
      f"({has_rationale/len(train)*100:.1f}%)")