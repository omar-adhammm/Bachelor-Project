import json
from pathlib import Path

folder = Path('outputs/results/lime_iou_analysis')
for f in folder.glob('*_lime_results.json'):
    with open(f) as fh:
        data = json.load(fh)
    print(f"{f.name}: n_examples={data.get('n_examples')}, "
          f"len(examples)={len(data.get('examples', []))}")
    if data.get('examples'):
        print(f"  First example keys: {list(data['examples'][0].keys())}")
        print(f"  First example: {data['examples'][0]}")
        break