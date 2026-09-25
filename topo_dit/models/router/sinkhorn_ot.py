"""Balanced entropic transport; returns row probabilities, not a square bistochastic matrix."""
import math
import torch


def sinkhorn_knopp(cost_matrix, epsilon=0.2, iters=30):
    if cost_matrix.ndim != 3 or min(cost_matrix.shape) < 1:
        raise ValueError('cost must have shape (B, N, K), with nonempty dimensions')
    if epsilon <= 0 or not math.isfinite(epsilon) or iters < 1:
        raise ValueError('epsilon must be finite and positive; iters must be positive')
    if not torch.isfinite(cost_matrix).all():
        raise ValueError('cost contains non-finite values')
    # float32 minimum even under mixed precision; retain float64 for gradcheck.
    cost = cost_matrix if cost_matrix.dtype == torch.float64 else cost_matrix.float()
    log_p = -cost / epsilon
    n, k = cost.shape[-2:]
    # P has total mass 1: row marginals 1/N and column marginals 1/K.
    for _ in range(iters):
        log_p = log_p - torch.logsumexp(log_p, -1, keepdim=True) - math.log(n)
        log_p = log_p - torch.logsumexp(log_p, -2, keepdim=True) - math.log(k)
    # A = N*P at convergence. Final row normalization is exact up to rounding;
    # column balance is approximate at finite iterations and must be monitored.
    return torch.softmax(log_p, dim=-1)


def marginal_errors(a):
    n, k = a.shape[-2:]
    return {'row_error': (a.sum(-1) - 1).abs().amax(),
            'column_relative_error': (a.sum(-2) / (n / k) - 1).abs().amax()}
