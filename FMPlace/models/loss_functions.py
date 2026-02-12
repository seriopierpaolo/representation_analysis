# loss_functions/batch_contrastive_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


#This loss function leads to gradient explosion, need more investigation
class BatchContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_embeds, pos_embeds):
        # Normalize for cosine similarity
        query_embeds = F.normalize(query_embeds, p=2, dim=1)
        pos_embeds = F.normalize(pos_embeds, p=2, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(query_embeds, pos_embeds.T) / self.temperature
        labels = torch.arange(len(query_embeds)).to(query_embeds.device)

        # Cross entropy loss over batch
        loss = F.cross_entropy(logits, labels)
        return loss



class TripletLoss(nn.Module):
    def __init__(self, margin=0.4):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negatives):
        """
        anchor: (D,) or (D) per sample if called in loop OR (B, D) batch
        positive: same shape as anchor
        negatives: (Nneg, D) or (B, Nneg, D)
        This implementation expects anchor/positive to be 1-D vectors (per-sample) or
        will also accept batch versions if you vectorize the call.
        We'll implement a version suitable for your per-sample usage below.
        """
        # If negatives is 2D (Nneg, D), compute distances to each neg
        # anchor and positive expected 1D tensors
        # squared distances (more stable)
        pos_dist_sq = (anchor - positive).pow(2).sum()
        # negatives: shape (Nneg, D)
        neg_dist_sq = (anchor.unsqueeze(0) - negatives).pow(2).sum(dim=1)  # (Nneg,)

        # compute hinge loss per negative: max(0, pos - neg + margin)
        losses = F.relu(pos_dist_sq - neg_dist_sq + self.margin)  # (Nneg,)
        # select hardest negative (max) as you already do
        if losses.numel() == 0:
            # fallback: no negatives (shouldn't happen), return zero
            return torch.tensor(0.0, device=anchor.device)
        return losses.mean()  # or losses.max() if you prefer hardest-only
