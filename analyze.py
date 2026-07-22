import json
import torch
import numpy as np
import sys
sys.path.append('.')
import torch.nn.functional as F
from models.hatebert_baseline import HateBERTBaseline, get_tokenizer

with open('data/raw/official/test.json', 'r') as f:
    raw = json.load(f)

CHECKPOINT = 'outputs/model_checkpoints/proposed_epoch_31_loss_0.6994.pt'
LABEL_NAMES = ['normal', 'offensive', 'hatespeech']
device = torch.device('cpu')

tokenizer = get_tokenizer()
model = HateBERTBaseline(num_labels=3)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

def predict_proba(texts):
    results = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors='pt',
                          truncation=True, max_length=128, padding=True)
            output = model(enc['input_ids'], enc['attention_mask'])
            probs = F.softmax(output['logits'], dim=-1).squeeze().numpy()
            results.append(probs)
    return np.array(results)

explicit_keywords = [
    'nigger','nigga','faggot','kike','spic','chink','retard',
    'kill','hate','stupid','idiot','bitch','fuck','shit','cunt',
    'whore','slut','bastard','pigs','scum','vermin','trash',
    'filth','disgusting','ape','monkey','savage','terrorist',
    'criminal','dirty','evil','demon','devil','disease',
    'parasite','plague','sheboon','niglet','muzrat','muzzies',
    'beaners','muslimes','negress','redneck','dykes','hoes',
    'hoe','ghetto','ching','chong'
]

print("Scanning implicit examples for correct predictions...")
print()

good = []
for i, ex in enumerate(raw):
    if ex['label'] not in ['hatespeech', 'offensive']:
        continue
    text = ex['text']
    if len(text) > 200 or len(text) < 15:
        continue
    text_lower = text.lower()
    has_explicit = any(kw in text_lower for kw in explicit_keywords)
    if has_explicit:
        continue
    rationale_mask = ex.get('rationale_mask', [])
    if not rationale_mask or sum(rationale_mask) == 0:
        continue
    # Must have selective rationale (not all tokens marked)
    tokens = text.split()
    if len(tokens) == 0:
        continue
    ratio = sum(rationale_mask) / len(tokens)
    if ratio > 0.7:  # skip if almost all tokens are rationale
        continue

    probs = predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    if pred_label == ex['label'] and confidence > 0.45:
        rationale_tokens = [t for t, m in zip(tokens, rationale_mask) if m == 1]
        good.append({
            'index': i,
            'text': text,
            'true_label': ex['label'],
            'pred_label': pred_label,
            'confidence': confidence,
            'probs': probs.tolist(),
            'rationale_tokens': rationale_tokens,
            'rationale_ratio': ratio
        })

good.sort(key=lambda x: -x['confidence'])
print(f"Found {len(good)} correctly predicted implicit examples")
print()
for idx, g in enumerate(good[:10]):
    print(f"{idx+1}. index={g['index']} | label={g['true_label']} | conf={g['confidence']:.3f}")
    print(f"   Text: {g['text']}")
    print(f"   Rationale tokens: {g['rationale_tokens']}")
    print(f"   Probs: normal={g['probs'][0]:.3f} offensive={g['probs'][1]:.3f} hatespeech={g['probs'][2]:.3f}")
    print()