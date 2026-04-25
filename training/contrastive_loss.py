# training/contrastive_loss.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config_loader import load_config

config = load_config()


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss with in-batch negatives.
    
    For a batch of (original, CF) pairs:
    - Pulls same-class examples together
    - Pushes different-class examples apart
    - Uses ALL in-batch relationships, not just pairs
    
    Based on: Khosla et al. (2020) Supervised Contrastive Learning
    and SimCSE (Gao et al. 2021)
    """

    def __init__(self, temperature: float = None):
        super().__init__()
        self.temperature = temperature or config["models"]["proposed"]["contrastive_temperature"]

    def forward(
        self,
        orig_embeddings: torch.Tensor,
        cf_embeddings:   torch.Tensor,
        orig_labels:     torch.Tensor,
        cf_labels:       torch.Tensor,
    ) -> torch.Tensor:

        batch_size = orig_embeddings.size(0)
        device     = orig_embeddings.device

        # Normalize embeddings
        orig_norm = F.normalize(orig_embeddings, dim=1)
        cf_norm   = F.normalize(cf_embeddings,   dim=1)

        # Stack all embeddings and labels
        all_emb    = torch.cat([orig_norm, cf_norm], dim=0)   # [2B, H]
        all_labels = torch.cat([orig_labels, cf_labels], dim=0) # [2B]

        n = 2 * batch_size

        # Compute similarity matrix — clamp to avoid extreme values
        sim = torch.matmul(all_emb, all_emb.T) / self.temperature  # [2B, 2B]
        sim = torch.clamp(sim, min=-50, max=50)  # prevent overflow

        # Mask diagonal (self-similarity)
        mask_self = torch.eye(n, dtype=torch.bool, device=device)

        # Positive pair mask: same label, not self
        labels_i = all_labels.unsqueeze(1)
        labels_j = all_labels.unsqueeze(0)
        mask_pos = (labels_i == labels_j) & ~mask_self

        # Skip if no positive pairs exist
        if not mask_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # For numerical stability — subtract max before exp (stable softmax)
        # Set self-similarity to very negative number
        sim_masked = sim.clone()
        sim_masked[mask_self] = -1e9

        # Compute log sum exp over all non-self pairs
        log_sum_exp = torch.logsumexp(sim_masked, dim=1)  # [2B]

        # For each anchor, compute mean log prob over positive pairs
        loss_per_anchor = torch.zeros(n, device=device)

        for i in range(n):
            pos_indices = mask_pos[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue
            # log prob for each positive pair
            log_probs = sim[i, pos_indices] - log_sum_exp[i]
            loss_per_anchor[i] = -log_probs.mean()

        # Average over anchors that have positives
        has_pos = mask_pos.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        valid_losses = loss_per_anchor[has_pos]
        # Final safety check
        valid_losses = torch.nan_to_num(valid_losses, nan=0.0)
        
        return valid_losses.mean()


class CFContrastiveLoss(nn.Module):
    """
    Simplified pairwise contrastive loss specifically for CF pairs.

    Directly pushes each (original, counterfactual) pair apart.
    More interpretable and easier to explain in a thesis.
    """

    def __init__(self, temperature: float = None, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature or config["models"]["proposed"]["contrastive_temperature"]
        self.margin      = margin

    def forward(
        self,
        orig_embeddings: torch.Tensor,
        cf_embeddings:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            orig_embeddings: [batch_size, hidden_size]
            cf_embeddings:   [batch_size, hidden_size]

        Returns:
            scalar loss — minimized when orig and CF embeddings are far apart
        """
        # Normalize
        orig_norm = F.normalize(orig_embeddings, dim=1)
        cf_norm   = F.normalize(cf_embeddings,   dim=1)

        # Cosine similarity between each original and its CF pair
        # Shape: [batch_size]
        pair_similarity = (orig_norm * cf_norm).sum(dim=1)

        # We want pairs to be DISSIMILAR — so loss = max(0, similarity - margin_target)
        # Target: similarity should be LOW (close to -1 or 0)
        # Loss increases when pairs are too similar
        loss = F.relu(pair_similarity + self.margin)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Final loss: L_total = L_CE + lambda * L_supervised_contrastive
    """

    def __init__(self):
        super().__init__()
        self.ce_loss          = nn.CrossEntropyLoss()
        self.contrastive_loss = SupervisedContrastiveLoss()
        self.lambda_weight    = config["models"]["proposed"]["contrastive_weight"]

    def forward(
        self,
        logits:          torch.Tensor,
        labels:          torch.Tensor,
        orig_embeddings: torch.Tensor,
        cf_embeddings:   torch.Tensor,
        orig_labels:     torch.Tensor,
        cf_labels:       torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:

        ce   = self.ce_loss(logits, labels)
        cont = self.contrastive_loss(
            orig_embeddings, cf_embeddings,
            orig_labels, cf_labels
        )
        total = ce + self.lambda_weight * cont

        return total, {
            "total":       total.item(),
            "ce":          ce.item(),
            "contrastive": cont.item(),
            "lambda":      self.lambda_weight,
        }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Contrastive Loss Smoke Test ===\n")

    batch_size  = 8
    hidden_size = 768   # HateBERT hidden size
    num_classes = 3

    # Fake embeddings
    orig_emb = torch.randn(batch_size, hidden_size)
    cf_emb   = torch.randn(batch_size, hidden_size)
    logits   = torch.randn(batch_size, num_classes)
    labels   = torch.randint(0, num_classes, (batch_size,))
    orig_labels = torch.tensor([1, 1, 2, 2, 1, 1, 2, 2], dtype=torch.long)
    cf_labels   = torch.zeros(batch_size, dtype=torch.long) 

    print("── CFContrastiveLoss ──")
    cf_loss = CFContrastiveLoss()
    loss = cf_loss(orig_emb, cf_emb)
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    print("  ✓ Non-negative")

    print("\n── SupervisedContrastiveLoss ──")
    sup_loss = SupervisedContrastiveLoss()
    loss = sup_loss(orig_emb, cf_emb, orig_labels, cf_labels)
    print(f"  Loss value: {loss.item():.4f}")
    assert loss.item() >= 0, "Loss should be non-negative"
    print("  ✓ Non-negative")

    print("\n── CombinedLoss ──")
    combined = CombinedLoss()
    total_loss, loss_dict = combined(
        logits, labels,
        orig_emb, cf_emb,
        orig_labels, cf_labels  # add these two arguments
    )
    print(f"  Total loss:       {loss_dict['total']:.4f}")
    print(f"  CE loss:          {loss_dict['ce']:.4f}")
    print(f"  Contrastive loss: {loss_dict['contrastive']:.4f}")
    print(f"  Lambda weight:    {loss_dict['lambda']}")
    assert total_loss.item() >= 0
    print("  ✓ All checks passed")

    print("\n── Gradient flow check ──")
    orig_emb.requires_grad_(True)
    cf_emb.requires_grad_(True)
    logits.requires_grad_(True)
    total_loss, _ = combined(logits, labels, orig_emb, cf_emb, orig_labels, cf_labels)
    total_loss.backward()
    assert orig_emb.grad is not None, "No gradient for orig_embeddings!"
    assert cf_emb.grad is not None,   "No gradient for cf_embeddings!"
    assert logits.grad is not None,   "No gradient for logits!"
    print("  ✓ Gradients flow correctly through all inputs")

    print("\ncontrastive_loss.py smoke test passed!")