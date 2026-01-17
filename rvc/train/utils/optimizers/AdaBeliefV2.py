import math
from typing import Dict, List, Tuple, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ("AdaBeliefV2",)


class AdaBeliefV2(Optimizer):
    """
    Реализация оптимизатора AdaBelief.
    
    AdaBelief адаптирует шаг обучения в зависимости от "веры" (belief) в направление градиента.
    Он объединяет скорость сходимости адаптивных методов (Adam) и способность к обобщению (SGD).
    
    Ключевые особенности этой реализации:
    1. Decoupled Weight Decay (как в AdamW) для лучшей регуляризации.
    2. Полная поддержка torch._foreach для максимальной скорости обучения.
    3. Опциональный AMSGrad для борьбы с нестабильностью в конце обучения.
    
    Arguments:
        params (iterable): итерируемый объект параметров для оптимизации или dict.
        lr (float, optional): коэффициент скорости обучения (default: 1e-4).
        betas (Tuple[float, float], optional): коэффициенты для вычисления бегущих средних градиента и его квадрата (default: (0.9, 0.999)).
        eps (float, optional): слагаемое для численной стабильности (default: 1e-8).
        weight_decay (float, optional): коэффициент затухания весов (default: 0).
        amsgrad (bool, optional): использовать вариант AMSGrad из статьи "On the Convergence of Adam and Beyond". Помогает устранить осцилляции лосса в конце обучения.
        foreach (bool, optional): использовать быстрые векторные операции PyTorch (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        foreach: bool = True,
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
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", True)
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Выполняет один шаг оптимизации."""
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
        """Медленная реализация step (по одному тензору) для обратной совместимости."""
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        amsgrad = group["amsgrad"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("AdaBelief does not support sparse gradients")

            # 1. Decoupled Weight Decay (применяем до обновления моментов)
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            state = self.state[p]

            # Инициализация состояния
            if len(state) == 0:
                state["step"] = 0
                # Экспоненциальное скользящее среднее градиентов
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Экспоненциальное скользящее среднее дисперсии (variance)
                state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    # Хранит максимальное значение дисперсии
                    state["max_exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg = state["exp_avg"]
            exp_avg_var = state["exp_avg_var"]

            state["step"] += 1
            step = state["step"]

            # 2. Обновление момента (Momentum)
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # 3. Обновление дисперсии (Variance)
            # AdaBelief использует разницу (grad - momentum) вместо просто grad^2 как в Adam.
            # Это позволяет делать большие шаги, когда градиент стабилен.
            grad_residual = grad - exp_avg
            exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

            # 4. Вычисление знаменателя
            if amsgrad:
                # AMSGrad берет максимум из истории дисперсий, предотвращая рост LR
                max_exp_avg_var = state["max_exp_avg_var"]
                torch.maximum(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                denom = max_exp_avg_var.add(eps).sqrt()
            else:
                denom = exp_avg_var.add(eps).sqrt()

            # 5. Вычисление поправок смещения (Bias Correction)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            # Математическая оптимизация: 
            # вместо деления тензора denom на скаляр sqrt(bc2), мы умножаем шаг lr на этот скаляр.
            # Это уменьшает количество операций над тензорами.
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            
            # 6. Обновление параметров
            # p = p - step_size * (momentum / denominator)
            p.addcdiv_(exp_avg, denom, value=-step_size)

    def _step_foreach(self, group: Dict) -> None:
        """Быстрая реализация step с использованием torch._foreach функций."""
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        amsgrad = group["amsgrad"]

        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_vars = []
        max_exp_avg_vars = []

        # Сбор списков тензоров для пакетной обработки
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)
            
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    state["max_exp_avg_var"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            
            exp_avgs.append(state["exp_avg"])
            exp_avg_vars.append(state["exp_avg_var"])
            if amsgrad:
                max_exp_avg_vars.append(state["max_exp_avg_var"])

        if not params_with_grad:
            return

        # Синхронизация шагов (берем шаг у первого параметра)
        state = self.state[params_with_grad[0]]
        state["step"] += 1
        step = state["step"]
        # Обновляем шаг у остальных для корректности состояния
        for p in params_with_grad[1:]:
            self.state[p]["step"] = step

        # 1. Weight Decay (векторизованно)
        if weight_decay != 0:
            torch._foreach_mul_(params_with_grad, 1 - lr * weight_decay)

        # 2. Обновление момента (Momentum)
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

        # 3. Обновление дисперсии (Variance) на основе разницы (grad - momentum)
        grad_residuals = torch._foreach_sub(grads, exp_avgs)
        torch._foreach_mul_(exp_avg_vars, beta2)
        torch._foreach_addcmul_(exp_avg_vars, grad_residuals, grad_residuals, value=1 - beta2)

        # 4. Обработка знаменателя (включая AMSGrad и Epsilon)
        if amsgrad:
            torch._foreach_maximum_(max_exp_avg_vars, exp_avg_vars)
            denom = torch._foreach_add(max_exp_avg_vars, eps)
        else:
            denom = torch._foreach_add(exp_avg_vars, eps)
        
        torch._foreach_sqrt_(denom)

        # 5. Вычисление размера шага (Step Size) с поправкой
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        # 6. Финальное обновление весов
        # params = params - step_size * (exp_avgs / denom)
        torch._foreach_addcdiv_(params_with_grad, exp_avgs, denom, value=-step_size)


def get_inverse_sqrt_scheduler(optimizer, warmup_epochs=15, last_epoch=-1):
    """
    Возвращает готовый планировщик типа "Inverse Square Root" (Noam Decay).
    
    Args:
        optimizer: Оптимизатор (AdaBelief или другой).
        warmup_epochs: Количество эпох для разогрева (линейный рост LR).
    """
    def lr_lambda(current_step):
        # current_step соответствует эпохе
        ep = current_step + 1 
        
        if ep < warmup_epochs:
            # Линейный разогрев: от почти 0 до 1.0
            return float(ep) / float(max(1, warmup_epochs))
        
        # Формула плавного вечного затухания
        return (warmup_epochs ** 0.5) / (ep ** 0.5)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
