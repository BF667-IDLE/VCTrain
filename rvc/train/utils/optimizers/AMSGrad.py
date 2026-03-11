from typing import Dict, List, Tuple
import math

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ("AMSGrad",)


class AMSGrad(Optimizer):
    """AMSGrad optimizer with optional Gradient Centralization.

    Implements the AMSGrad algorithm, a variant of Adam that uses the maximum
    of past squared gradients for the denominator. This version also supports
    decoupled weight decay and gradient centralization.

    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to denominator to improve numerical stability (default: 1e-8)
        weight_decay: decoupled weight decay coefficient (default: 0)
        amsgrad: whether to use the AMSGrad variant of Adam (default: True)
        foreach: if True, use torch._foreach implementations for speed (default: True)
        use_gc: if True, apply gradient centralization (default: False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = True,
        foreach: bool = True,
        use_gc: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            use_gc=use_gc,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", True)
            group.setdefault("amsgrad", True)
            group.setdefault("use_gc", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["foreach"]:
                self._step_foreach(group)
            else:
                self._step_single(group)

        return loss

    def _step_single(self, group: Dict) -> None:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        amsgrad = group["amsgrad"]
        use_gc = group["use_gc"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("AMSGrad does not support sparse gradients")

            # Gradient Centralization (if enabled)
            if use_gc and grad.dim() > 1:
                grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

            state = self.state[p]

            # Initialize state
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            if amsgrad:
                max_exp_avg_sq = state["max_exp_avg_sq"]

            state["step"] += 1
            step = state["step"]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            # Decoupled weight decay
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            # Update biased first moment estimate
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # Update biased second raw moment estimate
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # AMSGrad: maintain maximum of second moments
            if amsgrad:
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1
            p.addcdiv_(exp_avg, denom, value=-step_size)

    def _step_foreach(self, group: Dict) -> None:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        amsgrad = group["amsgrad"]
        use_gc = group["use_gc"]

        params_with_grad: List[Tensor] = []
        grads: List[Tensor] = []
        exp_avgs: List[Tensor] = []
        exp_avg_sqs: List[Tensor] = []
        max_exp_avg_sqs: List[Tensor] = []

        for p in group["params"]:
            if p.grad is None:
                continue

            params_with_grad.append(p)
            grad = p.grad

            # Gradient Centralization (if enabled)
            if use_gc and grad.dim() > 1:
                grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

            grads.append(grad)

            state = self.state[p]

            # Initialize state
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            if amsgrad:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

        if not params_with_grad:
            return

        # Update step counters (all parameters share the same global step)
        first_state = self.state[params_with_grad[0]]
        first_state["step"] += 1
        step = first_state["step"]
        for p in params_with_grad[1:]:
            self.state[p]["step"] = step

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decoupled weight decay
        if weight_decay != 0:
            torch._foreach_mul_(params_with_grad, 1 - lr * weight_decay)

        # Update biased first moment estimate
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

        # Update biased second raw moment estimate
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

        # AMSGrad: maintain maximum of second moments
        if amsgrad:
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
            denom = torch._foreach_sqrt(max_exp_avg_sqs)
            torch._foreach_div_(denom, math.sqrt(bias_correction2))
            torch._foreach_add_(denom, eps)
        else:
            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(denom, math.sqrt(bias_correction2))
            torch._foreach_add_(denom, eps)

        step_size = lr / bias_correction1
        updates = torch._foreach_div(exp_avgs, denom)
        torch._foreach_add_(params_with_grad, updates, alpha=-step_size)
