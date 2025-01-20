from typing import Iterable
from torch.optim import Optimizer
import torch
import math
import collections

class NewOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, betas_aggmo=[0.0, 0.9, 0.99], weight_decay=0):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            betas_aggmo=betas_aggmo,
            step=0,
            weight_decay=weight_decay
        )
        super(NewOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']
            epsilon = group['epsilon']
            lr = group['lr']
            step = group['step'] + 1
            group['step'] = step
            betas_aggmo = group['betas_aggmo']
            total_moments = len(betas_aggmo)
            weight_decay = group['weight_decay']

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                if weight_decay != 0:
                    grad.add_(weight_decay, param.data)

                if param not in self.state:
                    self.state[param] = {
                        'm': torch.zeros_like(param.data),
                        'v': torch.zeros_like(param.data),
                        'momentum_buffers': [torch.zeros_like(param.data) for _ in betas_aggmo]
                    }

                state = self.state[param]
                m = state['m']
                v = state['v']
                momentum_buffers = state['momentum_buffers']

                # Adam's moment updates
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)

                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)

                adam_update = lr * m_hat / (v_hat.sqrt() + epsilon)

                # AggMo update
                aggmo_update = torch.zeros_like(param.data)
                for beta, momentum_buffer in zip(betas_aggmo, momentum_buffers):
                    momentum_buffer.mul_(beta).add_(grad)
                    aggmo_update.add_(momentum_buffer)

                aggmo_update = aggmo_update / total_moments

                # Combined update
                param.data -= adam_update + lr * aggmo_update / total_moments

        return loss
