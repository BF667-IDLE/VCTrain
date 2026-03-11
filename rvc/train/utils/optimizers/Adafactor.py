from typing import Dict, List, Tuple, Optional
import math

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ("Adafactor",)


class Adafactor(Optimizer):
    """Adafactor optimizer with optional Gradient Centralization.

    Implements Adafactor algorithm, a memory-efficient adaptive method that factors
    the second-moment estimates for matrix-shaped parameters, reducing memory from
    O(n*m) to O(n+m). This version supports both factored (matrix) and unfactored
    (vector) modes, optional first-order momentum, and gradient centralization.

    The optimizer can operate in two modes:
        - Relative step mode (default): learning rate is computed internally based on
          step number and parameter scale (recommended for most cases)
        - Fixed lr mode: external learning rate is used (set `relative_step=False`)

    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: external learning rate (default: None, meaning use relative step mode)
        eps: regularization constants as tuple (eps1, eps2):
            eps1: for squared gradient (default: 1e-30)
            eps2: for parameter scale (default: 1e-3)
        clip_threshold: threshold for update clipping (default: 1.0)
        decay_rate: coefficient for running average decay (default: -0.8, meaning 1 - t^-0.8)
        beta1: coefficient for first moment (default: None, i.e., no momentum)
        weight_decay: decoupled weight decay coefficient (default: 0)
        scale_parameter: whether to scale lr by parameter RMS (default: True)
        relative_step: whether to use time-dependent learning rate (default: True)
        warmup_init: whether to use warmup initialization for lr (default: False)
        foreach: use torch._foreach for speed when possible (default: True)
        use_gc: enable Gradient Centralization (default: False)
    """

    def __init__(
        self,
        params,
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
        foreach: bool = True,
        use_gc: bool = False,
    ) -> None:
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step=True")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")
        if eps[0] <= 0.0 or eps[1] <= 0.0:
            raise ValueError(f"Invalid epsilon values: {eps}")
        if clip_threshold <= 0:
            raise ValueError(f"Invalid clip_threshold: {clip_threshold}")
        if beta1 is not None and (beta1 < 0.0 or beta1 >= 1.0):
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
            foreach=foreach,
            use_gc=use_gc,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", True)
            group.setdefault("use_gc", False)
            group.setdefault("beta1", None)
            group.setdefault("warmup_init", False)

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

    @staticmethod
    def _rms(tensor: Tensor) -> Tensor:
        """Compute root mean square of a tensor."""
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _get_lr(param_group: Dict, state: Dict) -> float:
        """Compute learning rate based on step and parameter scale."""
        if param_group["relative_step"]:
            # Time-dependent base learning rate
            if param_group["warmup_init"]:
                min_step = 1e-6 * state["step"]
            else:
                min_step = 1e-2
            lr_t = min(min_step, 1.0 / math.sqrt(state["step"]))

            # Scale by parameter RMS if requested
            param_scale = 1.0
            if param_group["scale_parameter"]:
                param_scale = max(param_group["eps"][1], state["RMS"])
            return param_scale * lr_t
        else:
            # Fixed external learning rate
            return param_group["lr"]

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row: Tensor, exp_avg_sq_col: Tensor) -> Tensor:
        """Approximate squared gradient from row/col factors (rank-1 factorization)."""
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def _step_single(self, group: Dict) -> None:
        eps1, eps2 = group["eps"]
        beta1 = group["beta1"]
        clip_threshold = group["clip_threshold"]
        weight_decay = group["weight_decay"]
        use_gc = group["use_gc"]
        decay_rate = group["decay_rate"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Adafactor does not support sparse gradients")

            # Gradient Centralization (if enabled)
            if use_gc and grad.dim() > 1:
                grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

            # Handle mixed precision
            orig_dtype = grad.dtype
            if grad.dtype in {torch.float16, torch.bfloat16}:
                grad = grad.float()
                p_fp32 = p.float()
            else:
                p_fp32 = p

            state = self.state[p]
            grad_shape = grad.shape
            factored = len(grad_shape) >= 2
            use_first_moment = beta1 is not None

            # State initialization
            if len(state) == 0:
                state["step"] = 0
                state["RMS"] = 0.0

                if use_first_moment:
                    state["exp_avg"] = torch.zeros_like(grad)

                if factored:
                    state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                    state["exp_avg_sq_col"] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:]
                    ).to(grad)
                else:
                    state["exp_avg_sq"] = torch.zeros_like(grad)
            else:
                # Move states to correct device/dtype
                if use_first_moment:
                    state["exp_avg"] = state["exp_avg"].to(grad)
                if factored:
                    state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                    state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                else:
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

            state["step"] += 1
            state["RMS"] = float(self._rms(p_fp32))
            lr_t = self._get_lr(group, state)

            # Decay parameter for second moment (time-dependent)
            beta2t = 1.0 - math.pow(state["step"], decay_rate)

            # Prepare update
            update = grad.square().add_(eps1)

            if factored:
                exp_avg_sq_row = state["exp_avg_sq_row"]
                exp_avg_sq_col = state["exp_avg_sq_col"]

                # Update row and column sums
                exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)

                # Reconstruct approximation of squared gradient
                update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                update.mul_(grad)
            else:
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                update = exp_avg_sq.rsqrt().mul_(grad)

            # Update clipping
            rms_update = self._rms(update)
            if rms_update > 0:
                update.div_((rms_update / clip_threshold).clamp_(min=1.0))

            update.mul_(lr_t)

            # Apply first moment if enabled
            if use_first_moment:
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(update, alpha=1 - beta1)
                update = exp_avg

            # Weight decay (decoupled)
            if weight_decay != 0:
                p_fp32.mul_(1 - lr_t * weight_decay)

            # Apply update
            p_fp32.add_(-update)

            # Copy back if we converted dtype
            if orig_dtype != grad.dtype:
                p.copy_(p_fp32)

    def _step_foreach(self, group: Dict) -> None:
        eps1, eps2 = group["eps"]
        beta1 = group["beta1"]
        clip_threshold = group["clip_threshold"]
        weight_decay = group["weight_decay"]
        use_gc = group["use_gc"]
        decay_rate = group["decay_rate"]

        params_with_grad: List[Tensor] = []
        grads: List[Tensor] = []
        exp_avgs: List[Tensor] = []
        exp_avg_sqs: List[Tensor] = []
        exp_avg_sq_rows: List[Tensor] = []
        exp_avg_sq_cols: List[Tensor] = []
        states: List[Dict] = []
        orig_dtypes: List[torch.dtype] = []

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Adafactor does not support sparse gradients")

            # Gradient Centralization
            if use_gc and grad.dim() > 1:
                grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

            # Handle mixed precision
            orig_dtypes.append(grad.dtype)
            if grad.dtype in {torch.float16, torch.bfloat16}:
                grad = grad.float()
                p_fp32 = p.float()
            else:
                p_fp32 = p

            params_with_grad.append(p_fp32)
            grads.append(grad)

            state = self.state[p]
            states.append(state)

            # State initialization
            grad_shape = grad.shape
            factored = len(grad_shape) >= 2
            use_first_moment = beta1 is not None

            if len(state) == 0:
                state["step"] = 0
                state["RMS"] = 0.0

                if use_first_moment:
                    state["exp_avg"] = torch.zeros_like(grad)

                if factored:
                    state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                    state["exp_avg_sq_col"] = torch.zeros(
                        grad_shape[:-2] + grad_shape[-1:]
                    ).to(grad)
                else:
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if use_first_moment:
                    exp_avgs.append(state["exp_avg"])
                if factored:
                    exp_avg_sq_rows.append(state["exp_avg_sq_row"])
                    exp_avg_sq_cols.append(state["exp_avg_sq_col"])
                else:
                    exp_avg_sqs.append(state["exp_avg_sq"])
            else:
                # Move states to correct device/dtype
                if use_first_moment:
                    state["exp_avg"] = state["exp_avg"].to(grad)
                    exp_avgs.append(state["exp_avg"])
                if factored:
                    state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                    state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    exp_avg_sq_rows.append(state["exp_avg_sq_row"])
                    exp_avg_sq_cols.append(state["exp_avg_sq_col"])
                else:
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)
                    exp_avg_sqs.append(state["exp_avg_sq"])

        if not params_with_grad:
            return

        # Update step counters
        first_state = states[0]
        first_state["step"] += 1
        step = first_state["step"]
        for state in states[1:]:
            state["step"] = step

        # Compute RMS for all parameters
        for i, (p, state) in enumerate(zip(params_with_grad, states)):
            state["RMS"] = float(self._rms(p))

        # Compute learning rates
        lr_tensors = []
        for state in states:
            lr_t = self._get_lr(group, state)
            lr_tensors.append(torch.tensor(lr_t, device=params_with_grad[0].device))

        # Decay parameter
        beta2t = 1.0 - math.pow(step, decay_rate)

        # Prepare updates
        updates = []
        update_tensors = []  # For gradient parts after factorization

        for i, (grad, state) in enumerate(zip(grads, states)):
            update = grad.square().add_(eps1)

            if len(exp_avg_sq_rows) > 0 and i < len(exp_avg_sq_rows):  # Factored case
                exp_avg_sq_row = exp_avg_sq_rows[i]
                exp_avg_sq_col = exp_avg_sq_cols[i]

                exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)

                update_factor = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                update = update_factor.mul(grad)
            elif len(exp_avg_sqs) > 0:  # Unfactored case
                exp_avg_sq = exp_avg_sqs[i]
                exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                update = exp_avg_sq.rsqrt().mul(grad)

            update_tensors.append(update)

        # Apply clipping to all updates
        for i, update in enumerate(update_tensors):
            rms_update = self._rms(update)
            if rms_update > 0:
                update.div_((rms_update / clip_threshold).clamp_(min=1.0))

        # Scale by learning rate
        for i, update in enumerate(update_tensors):
            update.mul_(lr_tensors[i])

        # Apply first moment if enabled
        if beta1 is not None and exp_avgs:
            for i, (exp_avg, update) in enumerate(zip(exp_avgs, update_tensors)):
                exp_avg.mul_(beta1).add_(update, alpha=1 - beta1)
                update_tensors[i] = exp_avg

        # Weight decay and parameter update
        for i, (p, update, lr_t) in enumerate(zip(params_with_grad, update_tensors, lr_tensors)):
            if weight_decay != 0:
                p.mul_(1 - lr_t.item() * weight_decay)
            p.add_(-update)

        # Copy back if we converted dtype
        for i, (p, orig_dtype) in enumerate(zip(params_with_grad, orig_dtypes)):
            if orig_dtype in {torch.float16, torch.bfloat16}:
                group["params"][i].copy_(p)
