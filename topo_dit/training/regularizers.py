import torch
from torch.nn import functional as F


def compute_cognitive_losses(assignments, elements):
    """Optional entropy/decorrelation penalties, not guarantees of object semantics."""
    a = assignments.float()
    entropy = -(a * a.clamp_min(1e-8).log()).sum(-1).mean()
    e = F.normalize(elements.float(), dim=-1)
    similarity = e @ e.transpose(1, 2)
    k = e.shape[1]
    mask = ~torch.eye(k, device=e.device, dtype=torch.bool)
    decorrelation = similarity[:, mask].square().mean() if k > 1 else similarity.sum() * 0
    return entropy, decorrelation
