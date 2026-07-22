import json

with open('outputs/results/lime_iou_analysis/proposed_lime_results.json') as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))
print("N examples:", len(data.get("examples", [])))
if data.get("examples"):
    print("First example keys:", list(data["examples"][0].keys()))
    print("First example:", data["examples"][0])