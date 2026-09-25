import torch
from torch import nn
from torch.nn import functional as F
from .sinkhorn_ot import sinkhorn_knopp


class SinkhornStratifier(nn.Module):
    """Balanced feature aggregation; no guarantee that slots correspond to objects."""
    def __init__(self, dim, num_elements, sinkhorn_iters=30, epsilon=0.2, routing='sinkhorn'):
        super().__init__()
        if routing not in ('sinkhorn', 'softmax', 'uniform'):
            raise ValueError('routing must be sinkhorn, softmax or uniform')
        self.num_elements, self.sinkhorn_iters = num_elements, sinkhorn_iters
        self.epsilon, self.routing = epsilon, routing
        self.domain_anchors = nn.Parameter(torch.randn(1, num_elements, dim) * 0.02)
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)

    def forward(self, grid_tokens):
        b, n, _ = grid_tokens.shape
        q = F.normalize(self.proj_q(self.domain_anchors).float(), dim=-1).expand(b, -1, -1)
        k = F.normalize(self.proj_k(grid_tokens).float(), dim=-1)
        scores = torch.bmm(k, q.transpose(1, 2))
        if self.routing == 'sinkhorn':
            a = sinkhorn_knopp(-scores, self.epsilon, self.sinkhorn_iters)
        elif self.routing == 'softmax':
            a = (scores / self.epsilon).softmax(-1)
        else:
            a = torch.full_like(scores, 1 / self.num_elements)
        weights = a.transpose(1, 2)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        elements = torch.bmm(weights.to(grid_tokens.dtype), grid_tokens)
        return elements, a
