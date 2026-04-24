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

    entry = {
        "id": post_id,
        "text": " ".join(item["post_tokens"]),
        "label": majority,
        "label_id": label_map.get(majority, -1),
        "annotators": item["annotators"],
        "rationales": item.get("rationales", [])
    }
    all_data.append(entry)

# Reproducible shuffle
random.seed(42)
random.shuffle(all_data)

# 80 / 10 / 10 split
n = len(all_data)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)

train = all_data[:n_train]
val   = all_data[n_train:n_train + n_val]
test  = all_data[n_train + n_val:]

for name, data in [("train", train), ("validation", val), ("test", test)]:
    with open(f"data/raw/{name}.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} examples → data/raw/{name}.json")

print("\nLabel distribution (train):")
for label, count in Counter(ex["label"] for ex in train).items():
    print(f"  {label}: {count}")

print("\nLabel distribution (validation):")
for label, count in Counter(ex["label"] for ex in val).items():
    print(f"  {label}: {count}")

print("\nLabel distribution (test):")
for label, count in Counter(ex["label"] for ex in test).items():
    print(f"  {label}: {count}")