import torch
import torch.optim as optim
import math

class SAM(optim.Optimizer):

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
    
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
    
        defaults = dict( rho = rho, **kwargs )
        super(SAM, self).__init__( params, defaults )
        self.base_optimizer = base_optimizer( self.param_groups, **kwargs )
        self.rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / ( grad_norm + 1e-12 )
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['e_w'] = p.grad * scale
                p.add_(self.state[p]['e_w'])  # ascent step

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])  # descent step

        self.base_optimizer.step()  # apply optimizer step

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        
        shared_device = self.param_groups[0]["params"][0].device
        
        norm = torch.norm(
            torch.stack( [
                p.grad.norm( p = 2 ).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ] ),
            p = 2
        )
        return norm

class NovoGrad(optim.Optimizer):
    """Implements NovoGrad algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0.98))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    Example:
        >>> model = ResNet()
        >>> optimizer = NovoGrad(model.parameters(), lr=1e-2, weight_decay=1e-5)
    """

    def __init__(self, params, lr=0.01, betas=(0.95, 0.98), eps=1e-8,
                 weight_decay=0,grad_averaging=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,weight_decay=weight_decay,grad_averaging = grad_averaging)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NovoGrad does not support sparse gradients')
                state = self.state[p]
                g_2 = torch.sum(grad ** 2)
                if len(state) == 0:
                    state['step'] = 0
                    state['moments'] = grad.div(g_2.sqrt() +group['eps']) + \
                                       group['weight_decay'] * p.data
                    state['grads_ema'] = g_2
                moments = state['moments']
                grads_ema = state['grads_ema']
                beta1, beta2 = group['betas']
                state['step'] += 1
                grads_ema.mul_(beta2).add_(1 - beta2, g_2)

                denom = grads_ema.sqrt().add_(group['eps'])
                grad.div_(denom)
                # weight decay
                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    grad.add_(decayed_weights)

                # Momentum --> SAG
                if group['grad_averaging']:
                    grad.mul_(1.0 - beta1)

                moments.mul_(beta1).add_(grad) # velocity

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.add_(-step_size, moments)

        return loss

class Lamb(optim.Optimizer):
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
        adam: bool = False,
        debias: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state['step'])
                    bias_correction /= 1 - beta1 ** state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)