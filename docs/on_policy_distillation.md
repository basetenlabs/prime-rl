# On-Policy Distillation

Prime-RL supports a built-in demonstration-conditioned self-distillation mode that replaces reward-derived advantages + PPO/GRPO-style loss with token-level KL distillation against an EMA teacher.

## Self-Distill Mode

Use the dedicated loss type:

```toml
[trainer]
max_async_level = 0

[trainer.loss]
type = "self_distill"
divergence = "reverse_kl"  # "forward_kl" | "reverse_kl" | "symmetric"
top_k = 64
ema_alpha = 0.01
loss_mask_prefix_tokens = 0
symmetric_mix = 0.5

[orchestrator]
max_async_level = 0
max_off_policy_steps = 0
```

In this mode:

- teacher inference servers are disabled (`teacher_gpu_ids`, `teacher_inference`, and `orchestrator.teacher_model` must not be used);
- training is strict on-policy (`trainer.max_async_level = 0`, `orchestrator.max_async_level = 0`);
- `orchestrator.max_off_policy_steps` is forced to `0`.

## Dataset Requirement

Self-distill mode requires OpenAI-style `messages` in the training dataset.

- `x` is the conversation up to and including the final user message before the final assistant message.
- `c` is the final assistant message.
- non-text or multimodal/structured message content is rejected in v1.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.loss.type` | `"default"` | Set to `"self_distill"` to enable demonstration-conditioned distillation mode. |
| `trainer.loss.divergence` | `"reverse_kl"` | Distillation divergence: `forward_kl`, `reverse_kl`, or `symmetric`. |
| `trainer.loss.top_k` | `64` | Student-selected Top-K for KL approximation. |
| `trainer.loss.ema_alpha` | `0.01` | EMA update factor for the teacher weights after each optimizer step. |
| `trainer.loss.loss_mask_prefix_tokens` | `0` | Masks first N generated tokens from distillation loss (per sequence). |
| `trainer.loss.symmetric_mix` | `0.5` | Mix coefficient for `symmetric` divergence. |

## Monitoring

Self-distill mode logs:

- `distill_kl`: per-token KL values used by the distillation objective;
- `distill_tokens`: number of distillation tokens in the microbatch.
