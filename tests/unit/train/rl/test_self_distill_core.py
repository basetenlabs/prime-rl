import importlib
import importlib.util

import torch


def _get_compute_topk_tail_distill_loss():
    from prime_rl.trainer.rl import loss as loss_module

    assert hasattr(loss_module, "compute_topk_tail_distill_loss"), (
        "Missing `compute_topk_tail_distill_loss` in `prime_rl.trainer.rl.loss`."
    )
    return loss_module.compute_topk_tail_distill_loss


def _get_apply_prefix_generated_mask():
    from prime_rl.trainer.rl import loss as loss_module

    assert hasattr(loss_module, "apply_prefix_generated_mask"), (
        "Missing `apply_prefix_generated_mask` in `prime_rl.trainer.rl.loss`."
    )
    return loss_module.apply_prefix_generated_mask


def _get_ema_update():
    assert importlib.util.find_spec("prime_rl.trainer.rl.ema") is not None, (
        "Missing module `prime_rl.trainer.rl.ema`."
    )
    ema_module = importlib.import_module("prime_rl.trainer.rl.ema")
    assert hasattr(ema_module, "ema_update_"), "Missing `ema_update_` in `prime_rl.trainer.rl.ema`."
    return ema_module.ema_update_


def _probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1).clamp_min(1e-30)


def _full_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, divergence: str) -> torch.Tensor:
    ps = _probs(student_logits)
    pt = _probs(teacher_logits)
    if divergence == "forward_kl":
        return (pt * (torch.log(pt) - torch.log(ps))).sum(dim=-1)
    if divergence == "reverse_kl":
        return (ps * (torch.log(ps) - torch.log(pt))).sum(dim=-1)
    raise ValueError(f"Unsupported divergence: {divergence}")


def _reference_topk_tail(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int,
    divergence: str,
    eps: float = 1e-8,
    select_from: str = "student",
) -> torch.Tensor:
    ps = _probs(student_logits)
    pt = _probs(teacher_logits)
    base = ps if select_from == "student" else pt

    topk_idx = torch.topk(base, k=top_k, dim=-1).indices
    ps_k = torch.gather(ps, dim=-1, index=topk_idx).clamp_min(eps)
    pt_k = torch.gather(pt, dim=-1, index=topk_idx).clamp_min(eps)

    s_k = ps_k.sum(dim=-1).clamp_min(eps)
    t_k = pt_k.sum(dim=-1).clamp_min(eps)
    s_tail = (1.0 - s_k).clamp_min(eps)
    t_tail = (1.0 - t_k).clamp_min(eps)

    if divergence == "forward_kl":
        top = (pt_k * (torch.log(pt_k) - torch.log(ps_k))).sum(dim=-1)
        tail = t_tail * (torch.log(t_tail) - torch.log(s_tail))
        return top + tail
    if divergence == "reverse_kl":
        top = (ps_k * (torch.log(ps_k) - torch.log(pt_k))).sum(dim=-1)
        tail = s_tail * (torch.log(s_tail) - torch.log(t_tail))
        return top + tail
    raise ValueError(f"Unsupported divergence: {divergence}")


def _reference_topk_renorm_only(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int,
    divergence: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    ps = _probs(student_logits)
    pt = _probs(teacher_logits)

    topk_idx = torch.topk(ps, k=top_k, dim=-1).indices
    ps_k = torch.gather(ps, dim=-1, index=topk_idx).clamp_min(eps)
    pt_k = torch.gather(pt, dim=-1, index=topk_idx).clamp_min(eps)

    ps_kn = ps_k / ps_k.sum(dim=-1, keepdim=True).clamp_min(eps)
    pt_kn = pt_k / pt_k.sum(dim=-1, keepdim=True).clamp_min(eps)

    if divergence == "forward_kl":
        return (pt_kn * (torch.log(pt_kn) - torch.log(ps_kn))).sum(dim=-1)
    if divergence == "reverse_kl":
        return (ps_kn * (torch.log(ps_kn) - torch.log(pt_kn))).sum(dim=-1)
    raise ValueError(f"Unsupported divergence: {divergence}")


def _compute_impl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int,
    divergence: str,
    symmetric_mix: float = 0.5,
) -> torch.Tensor:
    compute_topk_tail_distill_loss = _get_compute_topk_tail_distill_loss()
    return compute_topk_tail_distill_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        top_k=top_k,
        divergence=divergence,
        symmetric_mix=symmetric_mix,
    )


def test_forward_kl_matches_full_when_topk_is_vocab():
    torch.manual_seed(0)
    student_logits = torch.randn(2, 4, 7)
    teacher_logits = torch.randn(2, 4, 7)

    actual = _compute_impl(student_logits, teacher_logits, top_k=7, divergence="forward_kl")
    expected = _full_kl(student_logits, teacher_logits, divergence="forward_kl")
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_reverse_kl_matches_full_when_topk_is_vocab():
    torch.manual_seed(1)
    student_logits = torch.randn(2, 3, 8)
    teacher_logits = torch.randn(2, 3, 8)

    actual = _compute_impl(student_logits, teacher_logits, top_k=8, divergence="reverse_kl")
    expected = _full_kl(student_logits, teacher_logits, divergence="reverse_kl")
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_symmetric_matches_weighted_forward_and_reverse():
    torch.manual_seed(2)
    student_logits = torch.randn(2, 5, 9)
    teacher_logits = torch.randn(2, 5, 9)
    mix = 0.3

    sym = _compute_impl(student_logits, teacher_logits, top_k=4, divergence="symmetric", symmetric_mix=mix)
    fwd = _compute_impl(student_logits, teacher_logits, top_k=4, divergence="forward_kl")
    rev = _compute_impl(student_logits, teacher_logits, top_k=4, divergence="reverse_kl")
    expected = mix * fwd + (1.0 - mix) * rev
    assert torch.allclose(sym, expected, atol=1e-6, rtol=1e-6)


def test_topk_selection_is_student_based():
    student_logits = torch.tensor([[[4.0, 3.0, 0.2, 0.1, -1.0, -2.0]]], dtype=torch.float32)
    teacher_logits = torch.tensor([[[-2.0, -1.5, 0.1, 0.3, 3.2, 2.8]]], dtype=torch.float32)

    actual = _compute_impl(student_logits, teacher_logits, top_k=2, divergence="forward_kl")
    expected_student_topk = _reference_topk_tail(
        student_logits,
        teacher_logits,
        top_k=2,
        divergence="forward_kl",
        select_from="student",
    )
    expected_teacher_topk = _reference_topk_tail(
        student_logits,
        teacher_logits,
        top_k=2,
        divergence="forward_kl",
        select_from="teacher",
    )

    assert torch.allclose(actual, expected_student_topk, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(actual, expected_teacher_topk, atol=1e-6, rtol=1e-6)


def test_tail_correction_changes_result_vs_topk_renorm_only():
    student_logits = torch.tensor([[[1.4, 1.2, 0.7, 0.2, -0.4, -1.3]]], dtype=torch.float32)
    teacher_logits = torch.tensor([[[1.1, 0.9, 0.8, -0.1, -0.6, -1.4]]], dtype=torch.float32)

    actual = _compute_impl(student_logits, teacher_logits, top_k=2, divergence="reverse_kl")
    expected_topk_tail = _reference_topk_tail(student_logits, teacher_logits, top_k=2, divergence="reverse_kl")
    renorm_only = _reference_topk_renorm_only(student_logits, teacher_logits, top_k=2, divergence="reverse_kl")

    assert torch.allclose(actual, expected_topk_tail, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(actual, renorm_only, atol=1e-6, rtol=1e-6)


def test_apply_prefix_generated_mask_masks_first_generated_tokens():
    apply_prefix_generated_mask = _get_apply_prefix_generated_mask()
    loss_mask = torch.tensor([[False, True, True, True, True, False, True]], dtype=torch.bool)
    generated_mask = torch.tensor([[False, True, True, True, True, False, True]], dtype=torch.bool)

    actual = apply_prefix_generated_mask(
        loss_mask=loss_mask,
        generated_mask=generated_mask,
        loss_mask_prefix_tokens=2,
    )
    expected = torch.tensor([[False, False, False, True, True, False, True]], dtype=torch.bool)
    assert torch.equal(actual, expected)


def test_apply_prefix_generated_mask_counts_generated_positions_not_unmasked_positions():
    apply_prefix_generated_mask = _get_apply_prefix_generated_mask()
    loss_mask = torch.tensor([[False, True, False, True, True, False, True]], dtype=torch.bool)
    generated_mask = torch.tensor([[False, True, True, True, True, False, True]], dtype=torch.bool)

    actual = apply_prefix_generated_mask(
        loss_mask=loss_mask,
        generated_mask=generated_mask,
        loss_mask_prefix_tokens=3,
    )
    expected = torch.tensor([[False, False, False, False, True, False, True]], dtype=torch.bool)
    assert torch.equal(actual, expected)


def test_apply_prefix_generated_mask_all_masked_when_prefix_exceeds_generated_len():
    apply_prefix_generated_mask = _get_apply_prefix_generated_mask()
    loss_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)
    generated_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)

    actual = apply_prefix_generated_mask(
        loss_mask=loss_mask,
        generated_mask=generated_mask,
        loss_mask_prefix_tokens=10,
    )
    expected = torch.tensor([[False, False, False, False]], dtype=torch.bool)
    assert torch.equal(actual, expected)


def test_ema_update_matches_formula_single_step():
    ema_update_ = _get_ema_update()
    alpha = 0.2
    teacher = [torch.tensor([1.0, -2.0], dtype=torch.float32)]
    student = [torch.tensor([3.0, 5.0], dtype=torch.float32)]

    ema_update_(teacher_params=teacher, student_params=student, alpha=alpha)
    expected = (1.0 - alpha) * torch.tensor([1.0, -2.0]) + alpha * torch.tensor([3.0, 5.0])
    assert torch.allclose(teacher[0], expected, atol=1e-7, rtol=1e-7)


def test_ema_update_matches_closed_form_two_steps():
    ema_update_ = _get_ema_update()
    alpha = 0.1
    teacher = [torch.tensor([2.0, -1.0], dtype=torch.float32)]
    student_step_1 = [torch.tensor([4.0, 1.0], dtype=torch.float32)]
    student_step_2 = [torch.tensor([8.0, -5.0], dtype=torch.float32)]
    t0 = teacher[0].clone()

    ema_update_(teacher_params=teacher, student_params=student_step_1, alpha=alpha)
    ema_update_(teacher_params=teacher, student_params=student_step_2, alpha=alpha)

    expected = (
        ((1.0 - alpha) ** 2) * t0
        + (alpha * (1.0 - alpha)) * student_step_1[0]
        + alpha * student_step_2[0]
    )
    assert torch.allclose(teacher[0], expected, atol=1e-7, rtol=1e-7)
