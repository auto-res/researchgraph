ai_integratorv1_setting = {
    "objective": "I am researching Optimizers for fine-tuning LLM. The aim is to find a better Optimizer.",
    "arxiv_url": "https://arxiv.org/abs/1804.00325v3",
    "github_url": "https://github.com/AtheMathmo/AggMo",
    "method_template": """
from typing import Iterable
from torch.optim import Optimizer

class NewOptimizer(Optimizer):
    def __init__(self, params: Iterable,...):
        "parameter initialization"
    
    def step(self, closure: None = None) -> None:
        "processing details"

""",
    "base_method_code": """
from torch.optim import Optimizer

class Adam(Optimizer):
    def __init__(self, params: Iterable, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            step=0
        )
        super(Adam, self).__init__(params, defaults)

    def step(self, closure: None = None) -> None:
        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']
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
""",
    "base_method_text": """
Adam, or Adaptive Moment Estimation, is one of the most popular optimization algorithms used for training deep learning models. 
It builds upon the concept of stochastic gradient descent (SGD) but incorporates momentum to enhance the efficiency and stability of learning. 
Adam adapts the learning rate for each parameter by maintaining an exponentially decaying average of past gradients (first moment) and the squared gradients (second moment). 
This dual-moment approach allows Adam to handle sparse gradients and improves convergence. 
The first moment tracks the mean of the gradients, which helps in understanding the direction of movement, while the second moment approximates the uncentered variance, providing insight into the spread or scale of the gradients.
At each update step, Adam computes the moving averages of the gradient (\( m_t \)) and the squared gradient (\( v_t \)), both initialized at zero and updated using decay rates \( \beta_1 \) (commonly 0.9) and \( \beta_2 \) (commonly 0.999). 
These moving averages are then corrected for bias due to initialization, resulting in bias-corrected estimates \( \hat{m}_t \) and \( \hat{v}_t \). 
The parameters are updated by subtracting a fraction of the corrected gradient over the square root of the corrected second moment, adjusted by a small term \( \epsilon \) (often \( 10^{-8} \)) for numerical stability.
Adam's key advantages include faster convergence due to momentum and resilience to non-stationary data and noise in the gradient. 
Its adaptive learning rate mechanism makes it suitable for a wide range of problems, from computer vision to natural language processing, by tailoring updates to the specific behavior of each parameter.
""",
}
