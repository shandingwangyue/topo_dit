import torch
from torch.nn import functional as F
from .flow_matching import FlowMatcher
from .regularizers import compute_cognitive_losses
from ..models.router.sinkhorn_ot import marginal_errors


class TopoDiTTrainer:
    def __init__(self, model, optimizer, config, device='cpu'):
        self.model = model.to(device)
        self.optimizer, self.config, self.device = optimizer, config, device
        self.flow_matcher = FlowMatcher()

    def compute_loss(self, batch, phase=2):
        z1 = batch['latent'].to(self.device)
        condition = batch['condition'].to(self.device)
        if phase == 1:
            # Mask whole spatial-temporal tokens. Neither router nor renderer sees
            # the hidden target. Loss uses hidden positions only.
            b, _, t, h, w = z1.shape
            n = t*h*w
            if n < 2:
                raise ValueError('masked reconstruction needs at least two tokens')
            count = max(1, min(n-1, round(n * self.config.mask_ratio)))
            mask = torch.zeros(b, n, device=z1.device, dtype=torch.bool)
            selected = torch.rand(b, n, device=z1.device).argsort(-1)[:, :count]
            mask.scatter_(1, selected, True)
            mask = mask.reshape(b, 1, t, h, w)
            pred, a, e = self.model.reconstruct(z1.masked_fill(mask, 0))
            base = (pred-z1).square().masked_select(mask.expand_as(z1)).mean()
        elif phase in (2, 3):
            zt, target, time = self.flow_matcher.construct_flow_target(z1)
            pred, a, e = self.model(zt, time, condition)
            base = F.mse_loss(pred, target)
        else:
            raise ValueError('phase must be 1, 2 or 3')
        ent, ortho = compute_cognitive_losses(a, e)
        loss = base + self.config.lambda_ent * ent + self.config.lambda_ortho * ortho
        metrics = {'loss': loss.detach(), 'base_loss': base.detach(), 'entropy': ent.detach(),
                   'decorrelation': ortho.detach(), **{k:v.detach() for k,v in marginal_errors(a).items()}}
        return loss, metrics

    def train_epoch(self, dataloader, epoch, phase):
        self.model.train()
        # All relevant bottleneck modules adapt to noisy grids in FM; freezing a
        # clean-only router would impose an untested distribution mismatch.
        totals, count = {}, 0
        for batch in dataloader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = self.compute_loss(batch, phase)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss; update aborted')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip,
                                           error_if_nonfinite=True)
            self.optimizer.step()
            weight = batch['latent'].shape[0]
            count += weight
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * weight
        if count == 0:
            raise ValueError('empty dataloader')
        return {'epoch': epoch, 'phase': phase, **{k:v/count for k,v in totals.items()}}

    def train_phase_1_routing(self, dataloader, epoch):
        return self.train_epoch(dataloader, epoch, 1)

    def train_phase_2_dynamics(self, dataloader, epoch):
        return self.train_epoch(dataloader, epoch, 2)

    def train_phase_3_joint(self, dataloader, epoch):
        return self.train_epoch(dataloader, epoch, 3)
