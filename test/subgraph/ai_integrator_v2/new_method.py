import torch
from typing import Iterable
from torch.optim import Optimizer  # Please do not change this code

class NewOptimizer(Optimizer):  # Please do not change the name of the class “NewOptimizer”.
    def __init__(self, params: Iterable, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            step=0
        )
        super(NewOptimizer, self).__init__(params, defaults)

    def step(self, closure: None = None) -> None:
        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']  # Modified here to correct the error
            epsilon = group['epsilon']
            lr = group['lr']
            step = group['step'] + 1
            group['step'] = step

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                if param not in self.state:
                    self.state[param] = {'m': torch.zeros_like(param.data), 'v': torch.zeros_like(param.data)}

                m = self.state[param]['m']
                v = self.state[param]['v']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)

                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)

                param.data -= lr * m_hat / (v_hat.sqrt() + epsilon)