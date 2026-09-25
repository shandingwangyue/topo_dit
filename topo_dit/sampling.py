import torch


@torch.no_grad()
def euler_flow_matching_sampler(model, condition, T, H, W, steps=50, seed=0,
                                edit_index=None, edit_scale=0.0):
    if steps < 1 or min(T, H, W) < 1:
        raise ValueError('steps and latent dimensions must be positive')
    if condition.ndim != 2 or condition.shape[1] != model.cond_dim:
        raise ValueError('condition shape mismatch')
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    z = torch.randn(condition.shape[0], model.latent_channels, T, H, W,
                    device=condition.device, dtype=condition.dtype, generator=generator)
    was_training = model.training
    model.eval()
    try:
        for step in range(steps):
            time = torch.full((z.shape[0],), step / steps, device=z.device)
            velocity, _, _ = model(z, time, condition, edit_index=edit_index, edit_scale=edit_scale)
            z = z + velocity / steps
            if not torch.isfinite(z).all():
                raise FloatingPointError('non-finite Euler state')
    finally:
        model.train(was_training)
    return z
