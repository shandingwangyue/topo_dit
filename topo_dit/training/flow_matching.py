import torch


class FlowMatcher:
    """Independent Gaussian/data coupling; no minibatch optimal-transport coupling."""
    def construct_flow_target(self, z1, z0=None, t=None):
        z0 = torch.randn_like(z1) if z0 is None else z0
        t = torch.rand(z1.shape[0], device=z1.device) if t is None else t
        if z0.shape != z1.shape or t.shape != (z1.shape[0],):
            raise ValueError('noise or time shape mismatch')
        if not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any():
            raise ValueError('time outside [0, 1]')
        t_view = t.reshape(-1, *([1] * (z1.ndim - 1)))
        return (1 - t_view) * z0 + t_view * z1, z1 - z0, t
