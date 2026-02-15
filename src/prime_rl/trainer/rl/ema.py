from collections.abc import Iterable

import torch
from torch import Tensor


def clone_params(params: Iterable[Tensor]) -> list[Tensor]:
    return [param.detach().clone() for param in params]


def copy_params_(target_params: Iterable[Tensor], source_params: Iterable[Tensor]) -> None:
    target = list(target_params)
    source = list(source_params)
    if len(target) != len(source):
        raise ValueError(f"Parameter length mismatch: target={len(target)} source={len(source)}")

    with torch.no_grad():
        for tgt, src in zip(target, source):
            if tgt.shape != src.shape:
                raise ValueError(f"Parameter shape mismatch: target={tgt.shape} source={src.shape}")
            tgt.copy_(src)


def ema_update_(teacher_params: Iterable[Tensor], student_params: Iterable[Tensor], alpha: float) -> None:
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha}")

    teacher = list(teacher_params)
    student = list(student_params)
    if len(teacher) != len(student):
        raise ValueError(f"Parameter length mismatch: teacher={len(teacher)} student={len(student)}")

    with torch.no_grad():
        for t_param, s_param in zip(teacher, student):
            if t_param.shape != s_param.shape:
                raise ValueError(f"Parameter shape mismatch: teacher={t_param.shape} student={s_param.shape}")
            t_param.mul_(1.0 - alpha).add_(s_param, alpha=alpha)
