import yaml
import os

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    config = load_config()
    print("Config loaded successfully!")
    print(f"  Labels: {config['labels']['names']}")
    print(f"  HateBERT model: {config['models']['hatebert']['name']}")
    print(f"  Groq model: {config['api']['groq_model']}")