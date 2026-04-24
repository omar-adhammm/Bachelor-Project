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
    Supervised Contrastive Loss for CF pairs.

    For each (original, counterfactual) pair:
    - original is harmful (label 1 or 2)
    - counterfactual is normal (label 0)

    The loss pushes their embeddings APART in the representation space,
    forcing the model to learn semantic intent rather than surface patterns.

    Based on: Khosla et al. (2020) "Supervised Contrastive Learning"
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
        """
        Args:
            orig_embeddings: [batch_size, hidden_size] — embeddings of original harmful texts
            cf_embeddings:   [batch_size, hidden_size] — embeddings of counterfactuals
            orig_labels:     [batch_size] — labels of originals (1 or 2)
            cf_labels:       [batch_size] — labels of CFs (always 0)

        Returns:
            scalar loss value
        """
        batch_size = orig_embeddings.size(0)

        # Normalize embeddings to unit sphere
        orig_embeddings = F.normalize(orig_embeddings, dim=1)
        cf_embeddings   = F.normalize(cf_embeddings,   dim=1)

        # Stack all embeddings and labels together
        # Shape: [2 * batch_size, hidden_size]
        all_embeddings = torch.cat([orig_embeddings, cf_embeddings], dim=0)
        all_labels     = torch.cat([orig_labels, cf_labels],         dim=0)

        # Compute pairwise cosine similarity matrix
        # Shape: [2B, 2B]
        similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T) / self.temperature

        # Mask out self-similarity (diagonal)
        mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=all_embeddings.device)
        similarity_matrix = similarity_matrix.masked_fill(mask_self, float('-inf'))

        # Build positive pair mask — same label = positive pair
        # Shape: [2B, 2B]
        labels_row = all_labels.unsqueeze(0)  # [1, 2B]
        labels_col = all_labels.unsqueeze(1)  # [2B, 1]
        mask_positive = (labels_row == labels_col) & ~mask_self

        # For each anchor, compute log-softmax over all pairs
        log_prob = F.log_softmax(similarity_matrix, dim=1)

        # Mean log-prob over positive pairs only
        pos_count = mask_positive.sum(dim=1).float()
        
        # Skip anchors with no positive pairs
        has_positives = pos_count > 0
        
        if not has_positives.any():
            return torch.tensor(0.0, device=all_embeddings.device, requires_grad=True)

        loss = -(log_prob * mask_positive.float()).sum(dim=1) / torch.clamp(pos_count, min=1.0)
        
        # Only average over anchors that have positive pairs
        loss = loss[has_positives]
        
        # Clamp to avoid nan from edge cases
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        
        return loss.mean()


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
    Final loss used in the proposed model:
    L_total = L_CE + lambda * L_contrastive

    L_CE         = standard cross-entropy on all examples
    L_contrastive = CF pairwise contrastive loss
    lambda        = contrastive_weight from config (default 0.3)
    """

    def __init__(self):
        super().__init__()
        self.ce_loss          = nn.CrossEntropyLoss()
        self.contrastive_loss = CFContrastiveLoss()
        self.lambda_weight    = config["models"]["proposed"]["contrastive_weight"]

    def forward(
        self,
        # CE loss inputs
        logits:          torch.Tensor,   # [batch_size, num_classes]
        labels:          torch.Tensor,   # [batch_size]
        # Contrastive loss inputs
        orig_embeddings: torch.Tensor,   # [batch_size, hidden_size]
        cf_embeddings:   torch.Tensor,   # [batch_size, hidden_size]
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            loss_dict:  breakdown for logging
        """
        ce   = self.ce_loss(logits, labels)
        cont = self.contrastive_loss(orig_embeddings, cf_embeddings)

        total = ce + self.lambda_weight * cont

        loss_dict = {
            "total":       total.item(),
            "ce":          ce.item(),
            "contrastive": cont.item(),
            "lambda":      self.lambda_weight,
        }

        return total, loss_dict


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
    orig_labels = torch.tensor([1, 2, 1, 2, 1, 2, 1, 2], dtype=torch.long)
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
    total_loss, loss_dict = combined(logits, labels, orig_emb, cf_emb)
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
    total_loss, _ = combined(logits, labels, orig_emb, cf_emb)
    total_loss.backward()
    assert orig_emb.grad is not None, "No gradient for orig_embeddings!"
    assert cf_emb.grad is not None,   "No gradient for cf_embeddings!"
    assert logits.grad is not None,   "No gradient for logits!"
    print("  ✓ Gradients flow correctly through all inputs")

    print("\ncontrastive_loss.py smoke test passed!")