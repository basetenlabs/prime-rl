# Self-Distillation Replication Notes

Replicating [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) (Shen et al.) using Charlie's `feature/self-distill-mode` branch in prime-rl, with the paper's [tool-use dataset](https://github.com/idanshen/Self-Distillation/tree/main/data/tooluse_data).

## Paper vs Implementation — Key Differences

| Aspect | Paper (SDFT) | Charlie's Implementation |
|--------|-------------|--------------------------|
| KL scope | Full-vocabulary KL | Top-K (64) + tail approximation |
| KL direction | Forward KL (default) | Reverse KL (default) |
| Prefix skip | 3 tokens | 0 (configurable) |
| Entropy masking | Optional top-quantile | Not implemented |
| Off-policy support | IS correction | Strictly on-policy |
| EMA alpha | 0.01 | 0.01 |
| EMA update freq | Every step | Every step |
| Teacher context | Same template | Same template (extended for multi-turn) |

### Loss function

The paper computes KL divergence over the **full vocabulary** at each token position. Charlie's code approximates this with a top-K + tail decomposition: it takes the student's top-K tokens, gathers both student and teacher probabilities there, and lumps the remaining probability mass into a single "tail" bin. Cheaper, but coarser.

### Forward vs Reverse KL

- **Forward KL** `KL(teacher || student)` — mode-covering. The student must assign probability everywhere the teacher does. Produces broader distributions.
- **Reverse KL** `KL(student || teacher)` — mode-seeking. The student picks a subset of teacher-approved behavior and commits. Produces sharper distributions.

The paper defaults to forward KL. Charlie defaults to reverse KL. Both are supported via the `divergence` config field.

## Data Conversion

Paper format (`train_data.json`):
```json
{"prompt": "...", "golden_response": ["line1", "line2", ...]}
```

Charlie's expected format (OpenAI-style messages with `example_id`):
```json
{"example_id": 0, "messages": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "line1\nline2\n..."}
]}
```

The conversion joins `golden_response` lines with newlines into the final assistant message. The `self_distill_context.py` module then extracts the demonstration from that assistant message and builds the teacher prompt using the same template as the paper.

## Config for Paper-Faithful Replication

```toml
[loss]
type = "self_distill"
divergence = "forward_kl"
ema_alpha = 0.01
loss_mask_prefix_tokens = 3
top_k = 64
```

Paper hyperparameters for tool-use:
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Learning rate: `2e-5` (cosine schedule, 0.1 warmup ratio)
- Epochs: 1
- Batch size: 32 (via gradient accumulation)
- Max prompt/completion length: 1024

## Open Question

The top-K + tail KL approximation is the main gap from the paper's full-vocabulary KL. Options:
1. Run with top-K as-is and compare results
2. Add a full-vocab KL divergence option to `compute_topk_tail_distill_loss`

## Steps

1. Download `train_data.json` and `eval_data.json` from the paper repo
2. Write a conversion script to transform to Charlie's format
3. Write the training TOML config matching paper hyperparameters
4. Run training
5. Evaluate on tool-use eval set + prior capability benchmarks (HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval)
