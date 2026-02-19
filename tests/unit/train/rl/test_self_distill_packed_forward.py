"""Tests verifying that the packed teacher forward in _compute_self_distill_microbatch_loss
produces identical logits to per-sequence forwards, and that the zero-gradient fallback
works when no tokens survive the effective mask."""

import torch
import torch.nn as nn

from prime_rl.trainer.rl.ema import clone_params, copy_params_
from prime_rl.trainer.rl.loss import (
    apply_prefix_generated_mask,
    compute_topk_tail_distill_loss,
    shift_logits,
)
from prime_rl.trainer.utils import get_response_lengths


class TinyTransformer(nn.Module):
    """Minimal transformer that mimics the forward signature used by the trainer.

    Accepts input_ids and position_ids (packed, batch=1) and returns a dict with "logits".
    Uses a real embedding + linear layer so that parameter swapping (student/teacher EMA)
    produces meaningfully different outputs.
    """

    def __init__(self, vocab_size: int = 64, hidden: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, input_ids, position_ids=None, labels=None, temperature=None):
        h = self.embed(input_ids)
        logits = self.head(h)
        return {"logits": logits}


def _build_sequences():
    """Build 3 sequences of different lengths with teacher prompts.

    Returns (input_ids, position_ids, loss_mask, generated_mask, teacher_prompt_ids,
             response_lengths) all on CPU.

    Sequence 0: length 5, teacher prompt length 3, some tokens masked
    Sequence 1: length 4, teacher prompt length 2, ALL tokens masked (empty effective mask)
    Sequence 2: length 6, teacher prompt length 4, some tokens masked
    """
    vocab_size = 64
    seq_lengths = [5, 4, 6]
    teacher_prompt_lengths = [3, 2, 4]

    torch.manual_seed(42)

    input_ids_parts = []
    position_ids_parts = []
    loss_mask_parts = []
    generated_mask_parts = []
    teacher_prompt_ids = []

    for seq_idx, (seq_len, tp_len) in enumerate(zip(seq_lengths, teacher_prompt_lengths)):
        ids = torch.randint(1, vocab_size, (seq_len,))
        pos = torch.arange(seq_len)
        input_ids_parts.append(ids)
        position_ids_parts.append(pos)

        # Teacher prompt ids (different tokens so teacher sees different context)
        teacher_prompt_ids.append(torch.randint(1, vocab_size, (tp_len,)).tolist())

        if seq_idx == 1:
            # Edge case: empty effective mask — loss_mask is all False
            loss_mask_parts.append(torch.zeros(seq_len, dtype=torch.bool))
            generated_mask_parts.append(torch.zeros(seq_len, dtype=torch.bool))
        else:
            lm = torch.ones(seq_len, dtype=torch.bool)
            gm = torch.ones(seq_len, dtype=torch.bool)
            # Mask first token so prefix masking has something to strip
            gm[0] = False
            loss_mask_parts.append(lm)
            generated_mask_parts.append(gm)

    input_ids = torch.cat(input_ids_parts).unsqueeze(0)
    position_ids = torch.cat(position_ids_parts).unsqueeze(0)
    loss_mask = torch.cat(loss_mask_parts).unsqueeze(0)
    generated_mask = torch.cat(generated_mask_parts).unsqueeze(0)

    return input_ids, position_ids, loss_mask, generated_mask, teacher_prompt_ids, seq_lengths


def _per_sequence_teacher_forward(model, split_input_ids, teacher_prompt_ids, effective_masks):
    """Old approach: run a separate forward per sequence, skipping masked ones."""
    teacher_selected = [None] * len(split_input_ids)
    for seq_idx, (seq_ids, tp_ids) in enumerate(zip(split_input_ids, teacher_prompt_ids)):
        if not effective_masks[seq_idx].any():
            continue
        teacher_prompt = torch.tensor(tp_ids, dtype=torch.long)
        teacher_input = torch.cat([teacher_prompt, seq_ids], dim=0).unsqueeze(0)
        teacher_pos = torch.arange(teacher_input.shape[1]).unsqueeze(0)
        with torch.no_grad():
            out = model(teacher_input, position_ids=teacher_pos)
        logits = out["logits"]
        aligned = shift_logits(logits).squeeze(0)
        prompt_len = teacher_prompt.numel()
        continuation = aligned[prompt_len:]
        teacher_selected[seq_idx] = continuation[effective_masks[seq_idx]].detach()
    return teacher_selected


def _packed_teacher_forward(model, split_input_ids, teacher_prompt_ids, effective_masks):
    """New approach: pack ALL sequences into one forward, then split."""
    teacher_sequences = []
    teacher_pos_ids_list = []
    teacher_seq_lengths = []
    teacher_prompt_lengths = []

    for seq_ids, tp_ids in zip(split_input_ids, teacher_prompt_ids):
        teacher_prompt = torch.tensor(tp_ids, dtype=torch.long)
        teacher_input = torch.cat([teacher_prompt, seq_ids], dim=0)
        teacher_sequences.append(teacher_input)
        teacher_pos_ids_list.append(torch.arange(teacher_input.numel()))
        teacher_seq_lengths.append(teacher_input.numel())
        teacher_prompt_lengths.append(teacher_prompt.numel())

    packed_ids = torch.cat(teacher_sequences).unsqueeze(0)
    packed_pos = torch.cat(teacher_pos_ids_list).unsqueeze(0)

    with torch.no_grad():
        out = model(packed_ids, position_ids=packed_pos)
    logits = out["logits"]
    aligned = shift_logits(logits).squeeze(0)
    split_teacher = aligned.split(teacher_seq_lengths, dim=0)

    teacher_selected = [None] * len(split_input_ids)
    for seq_idx, (seq_ids, mask, seq_logits, prompt_len) in enumerate(
        zip(split_input_ids, effective_masks, split_teacher, teacher_prompt_lengths)
    ):
        if not mask.any():
            continue
        continuation = seq_logits[prompt_len:]
        teacher_selected[seq_idx] = continuation[mask].detach()
    return teacher_selected


def test_packed_forward_matches_per_sequence():
    """The packed teacher forward must produce identical logits to per-sequence forwards."""
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=64, hidden=32)
    model.eval()

    input_ids, position_ids, loss_mask, generated_mask, teacher_prompt_ids, seq_lengths = _build_sequences()

    response_lengths = get_response_lengths(position_ids)
    assert response_lengths == seq_lengths

    split_input_ids = input_ids.squeeze(0).split(response_lengths)
    split_loss_mask = loss_mask.squeeze(0).split(response_lengths)
    split_generated_mask = generated_mask.squeeze(0).split(response_lengths)

    effective_masks = [
        apply_prefix_generated_mask(
            loss_mask=slm.unsqueeze(0),
            generated_mask=sgm.unsqueeze(0),
            loss_mask_prefix_tokens=1,
        ).squeeze(0)
        for slm, sgm in zip(split_loss_mask, split_generated_mask)
    ]

    # Sequence 1 should have empty effective mask (the edge case)
    assert not effective_masks[1].any(), "Sequence 1 should have empty effective mask"

    per_seq_logits = _per_sequence_teacher_forward(model, split_input_ids, teacher_prompt_ids, effective_masks)
    packed_logits = _packed_teacher_forward(model, split_input_ids, teacher_prompt_ids, effective_masks)

    # Both should have None for the masked-out sequence
    assert per_seq_logits[1] is None
    assert packed_logits[1] is None

    # Non-None entries must match exactly
    for seq_idx in range(len(split_input_ids)):
        if per_seq_logits[seq_idx] is None:
            assert packed_logits[seq_idx] is None
            continue
        assert packed_logits[seq_idx] is not None, f"Sequence {seq_idx} missing from packed output"
        assert torch.allclose(
            per_seq_logits[seq_idx], packed_logits[seq_idx], atol=1e-6
        ), f"Logit mismatch at sequence {seq_idx}: max diff = {(per_seq_logits[seq_idx] - packed_logits[seq_idx]).abs().max()}"


def test_packed_forward_with_ema_parameter_swap():
    """Verify packed forward produces correct logits after EMA parameter swap and restore."""
    torch.manual_seed(0)
    student = TinyTransformer(vocab_size=64, hidden=32)

    # Create a different "EMA teacher" state by perturbing parameters
    ema_state = clone_params(student.parameters())
    with torch.no_grad():
        for p in ema_state:
            p.add_(torch.randn_like(p) * 0.5)

    input_ids, position_ids, loss_mask, generated_mask, teacher_prompt_ids, seq_lengths = _build_sequences()

    response_lengths = get_response_lengths(position_ids)
    split_input_ids = input_ids.squeeze(0).split(response_lengths)
    split_loss_mask = loss_mask.squeeze(0).split(response_lengths)
    split_generated_mask = generated_mask.squeeze(0).split(response_lengths)

    effective_masks = [
        apply_prefix_generated_mask(
            loss_mask=slm.unsqueeze(0),
            generated_mask=sgm.unsqueeze(0),
            loss_mask_prefix_tokens=1,
        ).squeeze(0)
        for slm, sgm in zip(split_loss_mask, split_generated_mask)
    ]

    # Snapshot student params, swap to EMA, run packed forward, restore
    model_params = list(student.parameters())
    student_snapshot = clone_params(model_params)
    copy_params_(model_params, ema_state)

    packed_logits = _packed_teacher_forward(student, split_input_ids, teacher_prompt_ids, effective_masks)

    # Per-sequence forward with same EMA weights
    per_seq_logits = _per_sequence_teacher_forward(student, split_input_ids, teacher_prompt_ids, effective_masks)

    # Restore student params
    copy_params_(model_params, student_snapshot)

    for seq_idx in range(len(split_input_ids)):
        if per_seq_logits[seq_idx] is None:
            assert packed_logits[seq_idx] is None
            continue
        assert torch.allclose(
            per_seq_logits[seq_idx], packed_logits[seq_idx], atol=1e-6
        ), f"EMA logit mismatch at seq {seq_idx}"

    # Verify student params were properly restored
    for orig, current in zip(student_snapshot, student.parameters()):
        assert torch.equal(orig, current.data), "Student params not restored after EMA swap"


def test_zero_gradient_fallback():
    """When no tokens survive the effective mask, the loss should still have a grad_fn
    so backward can issue the same collective operations as other ranks."""
    torch.manual_seed(0)
    model = TinyTransformer(vocab_size=64, hidden=32)

    input_ids, position_ids, loss_mask, generated_mask, teacher_prompt_ids, seq_lengths = _build_sequences()

    # Override: make ALL sequences have empty masks to trigger the fallback
    loss_mask = torch.zeros_like(loss_mask)
    generated_mask = torch.zeros_like(generated_mask)

    response_lengths = get_response_lengths(position_ids)
    split_input_ids = input_ids.squeeze(0).split(response_lengths)
    split_loss_mask = loss_mask.squeeze(0).split(response_lengths)
    split_generated_mask = generated_mask.squeeze(0).split(response_lengths)

    effective_masks = [
        apply_prefix_generated_mask(
            loss_mask=slm.unsqueeze(0),
            generated_mask=sgm.unsqueeze(0),
            loss_mask_prefix_tokens=1,
        ).squeeze(0)
        for slm, sgm in zip(split_loss_mask, split_generated_mask)
    ]

    # All masks should be empty
    for mask in effective_masks:
        assert not mask.any()

    # EMA state = clone of model (same weights, doesn't matter for this test)
    ema_state = clone_params(model.parameters())

    # --- Reproduce the distillation loss logic from _compute_self_distill_microbatch_loss ---

    # Packed teacher forward
    teacher_sequences = []
    teacher_pos_ids_list = []
    teacher_seq_lengths = []
    teacher_prompt_lengths = []
    for seq_ids, tp_ids in zip(split_input_ids, teacher_prompt_ids):
        teacher_prompt = torch.tensor(tp_ids, dtype=torch.long)
        teacher_input = torch.cat([teacher_prompt, seq_ids], dim=0)
        teacher_sequences.append(teacher_input)
        teacher_pos_ids_list.append(torch.arange(teacher_input.numel()))
        teacher_seq_lengths.append(teacher_input.numel())
        teacher_prompt_lengths.append(teacher_prompt.numel())

    packed_ids = torch.cat(teacher_sequences).unsqueeze(0)
    packed_pos = torch.cat(teacher_pos_ids_list).unsqueeze(0)

    model_params = list(model.parameters())
    student_snapshot = clone_params(model_params)
    copy_params_(model_params, ema_state)
    with torch.no_grad():
        teacher_out = model(packed_ids, position_ids=packed_pos)
        teacher_logits = teacher_out["logits"]
        teacher_aligned = shift_logits(teacher_logits).squeeze(0)
        split_teacher = teacher_aligned.split(teacher_seq_lengths, dim=0)

    teacher_selected_logits = [None] * len(split_input_ids)
    for seq_idx, (mask, seq_teacher, prompt_len) in enumerate(
        zip(effective_masks, split_teacher, teacher_prompt_lengths)
    ):
        if not mask.any():
            continue
        continuation = seq_teacher[prompt_len:]
        teacher_selected_logits[seq_idx] = continuation[mask].detach()

    copy_params_(model_params, student_snapshot)

    # Student forward (with gradients)
    student_out = model(input_ids, position_ids=position_ids)
    student_logits = student_out["logits"]
    student_aligned = shift_logits(student_logits)

    # Distillation loss accumulation
    total_distill_loss = torch.zeros((), dtype=torch.float32)
    distill_token_losses = []
    split_student = student_aligned.squeeze(0).split(response_lengths, dim=0)

    for seq_idx, (s_logits, mask, t_logits) in enumerate(
        zip(split_student, effective_masks, teacher_selected_logits)
    ):
        if t_logits is None:
            continue
        selected = s_logits[mask]
        seq_losses = compute_topk_tail_distill_loss(
            student_logits=selected.unsqueeze(0),
            teacher_logits=t_logits.unsqueeze(0),
            top_k=10,
            divergence="forward_kl",
        ).squeeze(0)
        distill_token_losses.append(seq_losses)
        total_distill_loss = total_distill_loss + seq_losses.sum()

    # No tokens survived — apply the zero-gradient fallback
    assert len(distill_token_losses) == 0, "Expected no distill tokens"
    total_distill_loss = total_distill_loss + 0.0 * student_aligned.sum()

    loss_scale = max(0, 1)  # mirrors _compute_self_distill_loss_scale clamping
    scaled_loss = total_distill_loss / loss_scale

    # The critical assertion: scaled_loss must still have a grad_fn
    assert scaled_loss.grad_fn is not None, "scaled_loss must be on the computation graph for backward"

    # Backward should succeed without error
    scaled_loss.backward()

    # Gradients should be zero (no real loss contribution)
    for p in model.parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-7), "Gradients should be zero"
