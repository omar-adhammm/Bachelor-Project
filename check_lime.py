import json

with open('outputs/results/lime_analysis/lime_results.json') as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))
print("Type:", type(data))

if isinstance(data, list):
    print("First entry keys:", list(data[0].keys()))
    print("First entry:", data[0])
elif isinstance(data, dict):
    first_key = list(data.keys())[0]
    print("First key:", first_key)
    print("First value type:", type(data[first_key]))
    if isinstance(data[first_key], dict):
        print("First value keys:", list(data[first_key].keys()))
    elif isinstance(data[first_key], list) and len(data[first_key]) > 0:
        print("First item in list:", data[first_key][0])