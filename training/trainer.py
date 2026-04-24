# training/trainer.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path

from configs.config_loader import load_config
from training.data_loader import (
    load_split,
    HateXplainDataset,
    ContrastiveHateDataset,
    load_cf_pairs,
    get_dataloaders,
    get_contrastive_dataloader,
)
from models.hatebert_baseline import HateBERTBaseline, get_tokenizer as get_tokenizer_baseline
from models.proposed_model import ProposedModel, get_tokenizer as get_tokenizer_proposed
from models.ablation_cf_only import AblationCFOnlyModel, get_tokenizer as get_tokenizer_ablation

config = load_config()


class ModelTrainer:
    """Unified trainer for all three models running simultaneously."""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.config = config
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.dataloaders = {}
        self.best_metrics = {}
        self.training_history = {}
        
        # Create output directory for results
        self.results_dir = Path(config["paths"]["results"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Device: {self.device}\n")

    def setup_models(self, model_name=None):
        """Initialize models — only load what's needed."""
        print("=== Setting up models ===\n")
        
        models_to_load = [model_name] if model_name and model_name != "all" else ["baseline", "proposed", "ablation"]
        
        if "baseline" in models_to_load:
            print("Loading HateBERT Baseline...")
            self.models["baseline"] = HateBERTBaseline(num_labels=3).to(self.device)
            self.training_history["baseline"] = {"train_loss": [], "val_loss": [], "val_acc": []}
            self.best_metrics["baseline"] = {"best_val_loss": float("inf"), "best_epoch": 0}

        if "proposed" in models_to_load:
            print("Loading ProposedModel (contrastive loss)...")
            self.models["proposed"] = ProposedModel(num_labels=3).to(self.device)
            self.training_history["proposed"] = {"train_loss": [], "val_loss": [], "val_acc": []}
            self.best_metrics["proposed"] = {"best_val_loss": float("inf"), "best_epoch": 0}

        if "ablation" in models_to_load:
            print("Loading AblationCFOnlyModel (no contrastive)...")
            self.models["ablation"] = AblationCFOnlyModel(num_labels=3).to(self.device)
            self.training_history["ablation"] = {"train_loss": [], "val_loss": [], "val_acc": []}
            self.best_metrics["ablation"] = {"best_val_loss": float("inf"), "best_epoch": 0}

        print("Models loaded.\n")

    def setup_data(self):
        """Load and prepare datasets for all models."""
        print("=== Setting up data ===\n")
        
        tokenizer = get_tokenizer_baseline()
        batch_size = config["models"]["hatebert"]["batch_size"]
        max_length = config["models"]["hatebert"]["max_length"]
        
        # Load standard HateXplain splits
        print("Loading HateXplain dataset...")
        train_examples = load_split("train")
        val_examples = load_split("validation")
        test_examples = load_split("test")
        
        # Dataloader for baseline and ablation (standard format)
        print("Creating standard dataloaders...")
        train_dataset = HateXplainDataset(train_examples, tokenizer, max_length)
        val_dataset = HateXplainDataset(val_examples, tokenizer, max_length)
        test_dataset = HateXplainDataset(test_examples, tokenizer, max_length)
        
        self.dataloaders["baseline_train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.dataloaders["baseline_val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.dataloaders["baseline_test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        print(f"  Train: {len(train_dataset)} examples, {len(self.dataloaders['baseline_train'])} batches")
        print(f"  Val:   {len(val_dataset)} examples, {len(self.dataloaders['baseline_val'])} batches")
        print(f"  Test:  {len(test_dataset)} examples, {len(self.dataloaders['baseline_test'])} batches\n")
        
        # Load CF pairs for proposed and ablation models
        print("Loading counterfactual pairs...")
        cf_path = f"{config['paths']['cf_pairs']}/train_cf_pairs.jsonl"
        
        if not os.path.exists(cf_path):
            raise FileNotFoundError(
                f"CF pairs not found at {cf_path}. "
                f"Please run: python counterfactuals/pipeline.py --split train"
            )
        
        cf_pairs = load_cf_pairs(cf_path)
        
        # Create contrastive dataset
        print("Creating contrastive dataloaders...")
        cf_dataset = ContrastiveHateDataset(cf_pairs, tokenizer, max_length)
        
        # Split CF pairs: use 90% for training, 10% for validation
        cf_train_size = int(0.9 * len(cf_dataset))
        cf_val_size = len(cf_dataset) - cf_train_size
        cf_train, cf_val = random_split(cf_dataset, [cf_train_size, cf_val_size])
        
        self.dataloaders["cf_train"] = DataLoader(
            cf_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.dataloaders["cf_val"] = DataLoader(
            cf_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        print(f"  CF Train: {len(cf_train)} pairs, {len(self.dataloaders['cf_train'])} batches")
        print(f"  CF Val:   {len(cf_val)} pairs, {len(self.dataloaders['cf_val'])} batches\n")

    def setup_optimizers(self):
        """Initialize optimizers and schedulers for all models."""
        print("=== Setting up optimizers ===\n")
        
        lr = float(config["models"]["hatebert"]["learning_rate"])
        warmup_steps = int(config["models"]["hatebert"]["warmup_steps"])
        
        for model_name, model in self.models.items():
            # Optimizer
            self.optimizers[model_name] = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=0.01,
            )
            
            
            print(f"  {model_name:12s}: LR={lr}, warmup_steps={warmup_steps}")
        
        print()

    def train_baseline_epoch(self, epoch):
        """Train baseline model for one epoch."""
        self.models["baseline"].train()
        total_loss = 0.0
        
        pbar = tqdm(self.dataloaders["baseline_train"], desc=f"Baseline Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizers["baseline"].zero_grad()
            
            output = self.models["baseline"](input_ids, attention_mask, labels)
            loss = output["loss"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models["baseline"].parameters(), 1.0)
            self.optimizers["baseline"].step()
            self.schedulers["baseline"].step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.dataloaders["baseline_train"])
        return avg_loss

    def train_contrastive_epoch(self, epoch, model_name):
        """Train proposed or ablation model using BOTH standard data and CF pairs."""
        self.models[model_name].train()
        total_loss = 0.0
        num_batches = 0

        # Create iterators for both dataloaders
        standard_iter = iter(self.dataloaders["baseline_train"])
        cf_iter       = iter(self.dataloaders["cf_train"])

        # Use the longer one as the loop length
        num_steps = max(
            len(self.dataloaders["baseline_train"]),
            len(self.dataloaders["cf_train"]),
        )

        desc = f"{model_name.capitalize()} Epoch {epoch+1}"
        pbar = tqdm(range(num_steps), desc=desc)

        for step in pbar:
            self.optimizers[model_name].zero_grad()
            batch_loss = torch.tensor(0.0, requires_grad=False)
            computed = False

            # ── Standard batch (CE loss on original examples) ──
            try:
                std_batch = next(standard_iter)
                input_ids      = std_batch["input_ids"].to(self.device)
                attention_mask = std_batch["attention_mask"].to(self.device)
                labels         = std_batch["label"].to(self.device)

                std_output = self.models[model_name](input_ids, attention_mask, labels)
                std_loss   = std_output["loss"]
                batch_loss = std_loss
                computed   = True
            except StopIteration:
                pass

            # ── CF batch (CE + contrastive loss) ──
            try:
                cf_batch = next(cf_iter)
                orig_input_ids      = cf_batch["orig_input_ids"].to(self.device)
                orig_attention_mask = cf_batch["orig_attention_mask"].to(self.device)
                orig_labels         = cf_batch["orig_label"].to(self.device)
                cf_input_ids        = cf_batch["cf_input_ids"].to(self.device)
                cf_attention_mask   = cf_batch["cf_attention_mask"].to(self.device)
                cf_labels           = cf_batch["cf_label"].to(self.device)

                cf_output = self.models[model_name].forward_pair(
                    orig_input_ids, orig_attention_mask, orig_labels,
                    cf_input_ids,   cf_attention_mask,   cf_labels,
                )

                if computed:
                    batch_loss = batch_loss + cf_output["loss"]
                else:
                    batch_loss = cf_output["loss"]
                    computed = True
            except StopIteration:
                pass

            if computed:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.models[model_name].parameters(), 1.0
                )
                self.optimizers[model_name].step()
                self.schedulers[model_name].step()

                total_loss  += batch_loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def evaluate(self, model_name, dataloader_key):
        """Evaluate a model on a validation/test set."""
        model = self.models[model_name]
        model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders[dataloader_key], desc=f"  Evaluating {model_name}..."):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                
                output = model(input_ids, attention_mask, labels)
                total_loss += output["loss"].item()
                
                probs = torch.softmax(output["logits"], dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.dataloaders[dataloader_key])
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        return avg_loss, accuracy

    def train(self, num_epochs=None, model_name=None):
        models_to_train = [model_name] if model_name and model_name != "all" else ["baseline", "proposed", "ablation"]
        if num_epochs is None:
            num_epochs = config["models"]["hatebert"]["epochs"]
        
        print(f"\n=== Training for {num_epochs} epochs ===\n")
        # ✅ Build correct schedulers NOW that we know epochs and dataloaders
        for name in models_to_train:
            if name == "baseline":
                steps_per_epoch = len(self.dataloaders["baseline_train"])
            else:
                steps_per_epoch = max(
                    len(self.dataloaders["baseline_train"]),
                    len(self.dataloaders["cf_train"]),
                )

            total_steps = steps_per_epoch * num_epochs

            self.schedulers[name] = get_linear_schedule_with_warmup(
                self.optimizers[name],
                num_warmup_steps=int(self.config["models"]["hatebert"]["warmup_steps"]),
                num_training_steps=total_steps,
            )

            print(f"{name}: steps/epoch={steps_per_epoch}, total_steps={total_steps}")
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---\n")
        
            for name in models_to_train:
                if name == "baseline":
                    loss = self.train_baseline_epoch(epoch)
                else:
                    loss = self.train_contrastive_epoch(epoch, name)
                self.training_history[name]["train_loss"].append(loss)
            
            print("\n--- Validation ---\n")
            for name in models_to_train:
                val_loss, val_acc = self.evaluate(name, "baseline_val")
                self.training_history[name]["val_loss"].append(val_loss)
                self.training_history[name]["val_acc"].append(val_acc)
                print(f"  {name:12s}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                if val_loss < self.best_metrics[name]["best_val_loss"]:
                    self.best_metrics[name]["best_val_loss"] = val_loss
                    self.best_metrics[name]["best_epoch"] = epoch
                    self.save_checkpoint(name, epoch, val_loss, val_acc)
        
        print("\n=== Training complete ===\n")
        self.print_summary()

    def save_checkpoint(self, model_name, epoch, val_loss, val_acc):
        """Save model checkpoint."""
        checkpoint_dir = Path(config["paths"]["checkpoints"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{model_name}_epoch_{epoch}_loss_{val_loss:.4f}.pt"
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.models[model_name].state_dict(),
            "optimizer_state_dict": self.optimizers[model_name].state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, checkpoint_path)
        
        print(f"    Saved checkpoint: {checkpoint_path}")

    def evaluate_on_test(self):
        """Final evaluation on test set."""
        print("\n=== Final Evaluation on Test Set ===\n")
        
        for model_name in self.training_history.keys():
            # Skip models that weren't trained this run
            if not self.training_history[model_name]["train_loss"]:
                continue
            test_loss, test_acc = self.evaluate(model_name, "baseline_test")
            print(f"  {model_name:12s}: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
            self.training_history[model_name]["test_loss"] = test_loss
            self.training_history[model_name]["test_acc"] = test_acc
        print()

    def print_summary(self):
        """Print training summary."""
        print("=== Final Summary ===\n")
        
        for model_name, hist in self.training_history.items():
            best = self.best_metrics[model_name]
            
            # Skip models that weren't trained this run
            if not hist["train_loss"]:
                continue
                
            print(f"{model_name.upper()}")
            print(f"  Best epoch:        {best['best_epoch']}")
            print(f"  Best val loss:     {best['best_val_loss']:.4f}")
            print(f"  Final train loss:  {hist['train_loss'][-1]:.4f}")
            print(f"  Final val loss:    {hist['val_loss'][-1]:.4f}")
            print(f"  Final val acc:     {hist['val_acc'][-1]:.4f}")
            if "test_acc" in hist:
                print(f"  Test acc:          {hist['test_acc']:.4f}")
            print()

    def save_results(self):
        """Save training history and results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"training_results_{timestamp}.json"
        
        results = {
            "timestamp": timestamp,
            "config": {
                "epochs": config["models"]["hatebert"]["epochs"],
                "learning_rate": config["models"]["hatebert"]["learning_rate"],
                "batch_size": config["models"]["hatebert"]["batch_size"],
                "contrastive_weight": config["models"]["proposed"]["contrastive_weight"],
            },
            "models": {},
        }
        
        for model_name in self.training_history.keys():
            # Skip models that weren't trained this run
            if not self.training_history[model_name]["train_loss"]:
                continue
            results["models"][model_name] = {
                "best_metrics": {
                    "best_epoch": int(self.best_metrics[model_name]["best_epoch"]),
                    "best_val_loss": float(self.best_metrics[model_name]["best_val_loss"]),
                },
                "history": {
                    "train_loss": [float(x) for x in self.training_history[model_name]["train_loss"]],
                    "val_loss": [float(x) for x in self.training_history[model_name]["val_loss"]],
                    "val_acc": [float(x) for x in self.training_history[model_name]["val_acc"]],
                },
            }
            if "test_loss" in self.training_history[model_name]:
                results["models"][model_name]["test_metrics"] = {
                    "test_loss": float(self.training_history[model_name]["test_loss"]),
                    "test_acc": float(self.training_history[model_name]["test_acc"]),
                }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}\n")


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all three models simultaneously")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: from config)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--model", type=str, default="all", choices=["baseline", "proposed", "ablation", "all"], help="Which model to train")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    trainer = ModelTrainer(device=device)
    
    # Setup
    trainer.setup_models(model_name=args.model)
    trainer.setup_data()
    trainer.setup_optimizers()
    
    # Train
    trainer.train(num_epochs=args.epochs, model_name=args.model)
    
    # Evaluate on test set
    trainer.evaluate_on_test()
    
    # Save results
    trainer.save_results()


if __name__ == "__main__":
    main()
