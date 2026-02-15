# Demonstration-Conditioned On-Policy Self-Distillation Plan

## 1) Goal

Integrate a new RL objective that replaces reward-derived advantages + GRPO/PPO loss with an on-policy self-distillation loss, while preserving:

- existing vLLM rollout generation path;
- existing trainer loop, optimizer, scheduler, distributed setup, logging, and checkpoint infrastructure;
- existing orchestrator scheduling and transport architecture.

This mode should plug into the current pipeline with minimal architectural disruption, but it is **not** a pure `LossFn` swap. It requires a dedicated trainer-step branch (separate teacher forward path, EMA parameter handling, and packed-sample metadata flow).

---

## 2) Locked Decisions (from this planning pass)

1. Demonstration source `c`:
- From the dataset's OpenAI `messages`.
- `c` is the final assistant message in the dataset conversation.

2. Student query/context `x`:
- Conversation up to and including the final user message (immediately before final assistant message).
- Supports single-turn and multi-turn message histories.

3. `{x}` rendering inside teacher template:
- Deterministic role transcript renderer (not chat-template text, not raw JSON).

4. Distillation divergences:
- Implement all three: `forward_kl`, `reverse_kl`, `symmetric`.
- Default: `reverse_kl`.

5. EMA:
- Use EMA teacher weights, update per optimizer step.
- Default `ema_alpha = 0.01`.

6. Top-K:
- Student-selected Top-K only.
- Include tail-mass correction (no renormalized Top-K-only loss).
- Default `top_k = 64` as a single global int.

7. Prefix artifact suppression:
- `loss_mask_prefix_tokens: int` applied per trajectory to generated-token positions.
- Default `0`.

8. Async policy for this mode:
- Strict on-policy required (`max_async_level = 0`).

9. Off-policy defense-in-depth:
- Also set `orchestrator.max_off_policy_steps = 0` in this mode (not strictly required when `max_async_level=0`, but prevents accidental drift).

---

## 3) Current Code Path (Integration Surface)

### Orchestrator side

- Rollout generation: `src/prime_rl/orchestrator/scheduler.py`
- Main loop and rollout -> `TrainingSample`: `src/prime_rl/orchestrator/orchestrator.py`
- Interleaving/multi-turn flattening: `src/prime_rl/orchestrator/trajectories.py`
- Optional teacher logprobs RPC path (legacy): `src/prime_rl/orchestrator/utils.py::compute_teacher_logprobs`
- Config: `src/prime_rl/orchestrator/config.py`

### Transport / packing

- Schemas: `src/prime_rl/transport/types.py`
- Batch prep and packing: `src/prime_rl/trainer/batch.py`, `src/prime_rl/trainer/rl/packer.py`
- Dataloader tensorization: `src/prime_rl/trainer/rl/data.py`

### Trainer side

- Main RL train loop: `src/prime_rl/trainer/rl/train.py`
- Existing loss wiring: `src/prime_rl/trainer/rl/loss.py`
- Model forward wrapper: `src/prime_rl/trainer/model.py`
- Trainer config: `src/prime_rl/trainer/rl/config.py`
- Checkpoint manager: `src/prime_rl/trainer/ckpt.py`

### Top-level config validation

- Shared RL config and cross-module validators: `src/prime_rl/rl.py`

---

## 4) Scope and Non-Goals

## In scope

- New built-in loss mode for self-distillation.
- Dataset-conditioned teacher context construction (`CtxT(x,c)`).
- EMA teacher weights in trainer.
- Teacher scoring on student trajectories (no teacher sampling).
- Top-K + tail-corrected divergence.
- Prefix-token masking for artifact suppression.
- Unit tests for core math + EMA behavior.

## Out of scope

- New rollout generator.
- New trainer loop framework.
- New checkpoint system.
- New eval harness.
- Replacing existing transport backends.

---

## 5) Required Semantics

## 5.1 Context builders

### Student context `CtxS(x)`

Student rollout remains the existing environment-driven prompt path. No new sampling path is added.

### Teacher context `CtxT(x,c)`

For each sample, construct with exact semantics:

```text
<Question>
{x}
This is an example for a response to the question:
<Demonstration>
{c}
Now answer with a response of your own, including the thinking process:
```

`x` is a deterministic transcript of messages up to final user message.
`c` is final assistant message from dataset example.

Deterministic transcript format for `x` (explicit spec):

```text
<USER>
{content}
</USER>
<ASSISTANT>
{content}
</ASSISTANT>
<TOOL>
{content}
</TOOL>
```

Rules:
- Preserve original message order.
- Include only textual content in v1.
- If any message content is non-text / multimodal / structured tool payload, fail fast with explicit `example_id` and role index.

## 5.2 Teacher scoring

- No teacher generation.
- Re-score the same student continuation tokens under teacher-conditioned context.

## 5.3 Prefix loss masking

- Add `loss_mask_prefix_tokens`.
- For each sequence, zero/ignore the first N generated-token positions from distillation loss.
- Apply over generated-token order, not raw token index.

## 5.4 EMA update

- Initialize teacher from student at startup (and on missing resume state).
- After each optimizer step:

`teacher = (1 - alpha) * teacher + alpha * student`

- Teacher forward is `no_grad`.

## 5.5 Top-K + tail correction

- Top-K must be selected by student distribution only.
- Tail correction must be explicit.
- No Top-K renormalization-only shortcut.

---

## 6) Data Contracts to Add

Extend transport sample payloads so trainer can compute teacher logits from a teacher prompt:

- Add teacher prompt token IDs to `TrainingSample` and `MicroBatch`.
- Keep fields optional/defaulted to preserve msgspec decode compatibility for old data.

Candidate fields:

- `TrainingSample.teacher_prompt_ids: list[int] | None = None`
- `MicroBatch.teacher_prompt_ids: list[list[int]] | None = None` (one teacher prompt per packed sample/sequence, in the same order as sequence splits derived from `position_ids`)
- `MicroBatch.generated_mask: list[bool]` (token-aligned; true for generated completion-token positions, false otherwise; used for prefix-token suppression)

No changes required to rollout generation payloads themselves.

---

## 7) File-by-File Implementation Plan

## 7.1 Config: add new loss mode

File: `src/prime_rl/trainer/rl/config.py`

Add a new discriminated loss config, e.g.:

- `type: Literal["self_distill"]`
- `divergence: Literal["forward_kl", "reverse_kl", "symmetric"] = "reverse_kl"`
- `top_k: int = 64`
- `ema_alpha: float = 0.01`
- `loss_mask_prefix_tokens: int = 0`
- `symmetric_mix: float = 0.5`

Validation:

- `top_k >= 1`
- `0 < ema_alpha <= 1`
- `loss_mask_prefix_tokens >= 0`
- `0 <= symmetric_mix <= 1`

Add trainer-level validator:

- in self-distill mode, fused LM head must be disabled (we need logits path).
- implement by **modifying existing** `auto_setup_fused_lm_head_chunk_size` (same validator), not by adding a second override validator:
  - if `loss.type == "self_distill"` and chunk size is `"auto"`, set `"disabled"`;
  - otherwise preserve current behavior (`"auto" -> 2048`).
- add v1 validator: reject `max_concurrent_runs > 1` when `loss.type == "self_distill"` (single-run only for first integration).

## 7.2 Shared RL config validation

File: `src/prime_rl/rl.py`

Add validator(s):

- if `trainer.loss.type == "self_distill"`, enforce strict on-policy:
  - `trainer.max_async_level == 0`
  - `orchestrator.max_async_level == 0`
- set / enforce `orchestrator.max_off_policy_steps == 0` for defense-in-depth.
- reject incompatible teacher-server assumptions for this mode (legacy `teacher_tau`-based teacher path should not be silently mixed).
- explicitly enforce for self-distill mode:
  - `orchestrator.teacher_model is None`
  - `teacher_inference is None` / no teacher inference server startup path.
- fix latent `teacher_tau` attribute access bug by guarding in `rl.py`:
  - `validate_teacher_model()` must only read `teacher_tau` for default `LossConfig`.
  - process startup warning branch must also guard `teacher_tau` access similarly.

## 7.3 Orchestrator context extraction and teacher prompt tokenization

Files:

- `src/prime_rl/orchestrator/orchestrator.py`
- new helper module: `src/prime_rl/orchestrator/self_distill_context.py`

Implementation:

1. Build fast lookup from sampled dataset examples:
- map `example_id -> raw dataset example` from `train_dataset`.

2. For each produced `TrainingSample`, derive:
- `x_messages`: all messages up to final user turn.
- `c_message`: final assistant message.
- deterministic transcript string for `x`.
- teacher prompt string via required template.

3. Tokenize teacher prompt once per training sample with orchestrator tokenizer.

4. Attach `teacher_prompt_ids` to each `TrainingSample`.

5. Disable legacy orchestrator-side `compute_teacher_logprobs` call for self-distill mode.

Failure behavior:

- Fail fast if required `messages` schema is missing or malformed for any sampled example.
- Include `example_id` in raised error.

## 7.4 Transport + packer + data loader propagation

Files:

- `src/prime_rl/transport/types.py`
- `src/prime_rl/trainer/batch.py`
- `src/prime_rl/trainer/rl/packer.py`
- `src/prime_rl/trainer/rl/data.py`

Implementation:

- Carry `teacher_prompt_ids` from `TrainingSample` through `MicroBatch`.
- In packed microbatches, store as `list[list[int]]` (per packed sample), not flat token-aligned list.
- Preserve per-sample ordering so `teacher_prompt_ids[i]` aligns with sequence `i` recovered from `position_ids` reset boundaries.
- In `packed_samples_into_micro_bs`, merge `teacher_prompt_ids` with `append(...)` (one prompt per sample), **not** `extend(...)`.
- Ensure truncation/padding logic for token-aligned fields does not mutate per-sample prompt lists.
- Tensorize / keep list form in dataloader as needed for per-sequence teacher forward construction.
- Add token-aligned `generated_mask` to `MicroBatch`:
  - source in `prepare_sample`: `generated_mask = [False] * len(prompt_ids) + completion_mask`
  - pack with existing token-aligned merge semantics (`extend`)
  - pad with `False`
  - pass to trainer dataloader tensors.

## 7.5 Trainer EMA teacher + distillation loss branch

Files:

- `src/prime_rl/trainer/rl/train.py`
- `src/prime_rl/trainer/rl/loss.py`
- new helper: `src/prime_rl/trainer/rl/ema.py`

Implementation:

1. Add self-distill branch in training step:
- **do not route through existing `LossInputs` / `compute_loss` scalar-logprob path**.
- add a dedicated branch in `train.py` for `self_distill` that consumes full-vocab logits.
- still compute student outputs on rollout tokens.
- compute teacher outputs on independently constructed teacher inputs under `no_grad`.
- teacher input construction is per sequence:
  - `teacher_input_ids = teacher_prompt_ids + continuation_ids`
  - `teacher_position_ids = arange(len(teacher_input_ids))`
  - align teacher logits to completion token positions only.
- sequence budget handling:
  - if `len(teacher_input_ids) > model.seq_len`, fail fast in v1 with explicit run/step/example identifiers and lengths.
  - (optional future enhancement: truncation policy config).
- teacher forward batching strategy (v1 decision):
  - run teacher forward **per sequence** (no padded teacher batch) for correctness/simplicity first;
  - revisit padded batching after correctness/perf baseline is established.

2. Add EMA lifecycle:
- create EMA shadow state at startup.
- EMA state must follow FSDP sharded parameter layout (local shards / DTensor-compatible storage).
- implement swap-in/swap-out context for teacher forward:
  - save current local shards;
  - copy EMA local shards into model params;
  - run teacher forward under `no_grad`;
  - restore original student shards.
- update EMA post-optimizer step on local shards.

3. Add distillation loss function:
- divergence modes: forward/reverse/symmetric;
- student-topK index selection;
- tail correction term;
- prefix generated-token masking.
- compute normalization (`loss_scale`) by effective distill token count:
  - compute once per step in self-distill branch **before** micro-batch loop;
  - count generated tokens that remain after prefix masking across local microbatches (using `generated_mask` + `loss_mask_prefix_tokens`);
  - clamp to minimum 1 to avoid divide-by-zero.

4. Keep existing `default` and `custom` loss modes unchanged.

5. Memory-aware execution strategy:
- avoid holding both student and teacher full-vocab logits simultaneously.
- process per sequence (or small chunks of sequences) for distillation metric/loss accumulation.
- explicitly clear teacher logits after use.
- add a conservative v1 guardrail in config/docs for seq_len and expected memory footprint when fused LM head is disabled.

## 7.6 Checkpoint / resume integration

File: `src/prime_rl/trainer/ckpt.py`

Implementation:

- store EMA state as sidecar local-shard tensors alongside trainer checkpoint (e.g. rank-local `ema.pt` under trainer checkpoint path) using `torch.save` / `torch.load`, not inside DCP `AppState`;
- load EMA on resume;
- if missing (old ckpt), initialize from current student and log once.

## 7.7 Docs and config examples

Files:

- `docs/on_policy_distillation.md`
- `docs/bring-your-own-algorithms.md`
- add debug config example under `configs/debug/rl/` for self-distill mode

Note: update existing docs only; do not add new markdown docs for v1.

Update content:

- new loss type and parameters;
- strict on-policy requirement;
- dataset `messages` requirement for `x` and `c`.

---

## 8) Distillation Math (Core Logic)

Let `p_s` be student next-token distribution and `p_t` teacher next-token distribution at each generated position.
Let `K` be Top-K token indices selected by `p_s` only.

Define:

- `S_K = sum_{i in K} p_s(i)`
- `T_K = sum_{i in K} p_t(i)`
- `S_tail = 1 - S_K`
- `T_tail = 1 - T_K`

## Forward KL (`T || S`)

`KL(T||S) = sum_i p_t(i) * (log p_t(i) - log p_s(i))`

Top-K + tail approximation:

- `sum_{i in K} p_t(i) * (log p_t(i) - log p_s(i))`
- `+ T_tail * (log T_tail - log S_tail)`

## Reverse KL (`S || T`)

`KL(S||T) = sum_i p_s(i) * (log p_s(i) - log p_t(i))`

Top-K + tail approximation:

- `sum_{i in K} p_s(i) * (log p_s(i) - log p_t(i))`
- `+ S_tail * (log S_tail - log T_tail)`

## Symmetric

`L_sym = mix * KL(T||S) + (1 - mix) * KL(S||T)`

with `mix = symmetric_mix`.

Numerics:

- clamp tail masses with small epsilon before `log`.
- if mask is empty, loss should be zero and metrics finite.

---

## 9) Execution Order

1. Add config schema + validations.
2. Add transport schema fields and propagation plumbing.
3. Implement orchestrator context extraction/tokenization and attach `teacher_prompt_ids`.
4. Implement trainer self-distill math (pure functions) and EMA helper.
5. Wire trainer branch and logging metrics.
6. Add checkpoint EMA persistence.
7. Add docs + debug config.
8. Run targeted unit tests, then broader suite.

---

## 10) Risks and Mitigations

1. Missing/heterogeneous dataset `messages` schema.
- Mitigation: explicit validation on first batch; hard fail with actionable message.

2. Fused LM head path hides logits needed for Top-K divergence.
- Mitigation: enforce fused-lm-head disabled in self-distill mode.

3. Sequence packing and per-sample teacher prompt alignment bugs.
- Mitigation: strict shape assertions and targeted packing tests.

4. Resume inconsistency for EMA state.
- Mitigation: checkpoint EMA and deterministic fallback init.

5. Async/off-policy mismatch with objective assumptions.
- Mitigation: enforce `max_async_level=0` and `max_off_policy_steps=0` in self-distill mode.

6. Latent config/runtime crash due `teacher_tau` attribute access on non-default loss configs.
- Mitigation: guard access with `isinstance(config.trainer.loss, LossConfig)` in both validator and startup warning branches.

7. Teacher-input overflow against `model.seq_len`.
- Mitigation: explicit per-sequence length check and fail-fast error in v1.

8. Full-vocab logit memory pressure when fused LM head is disabled.
- Mitigation: dedicated self-distill training branch, per-sequence accumulation, and no simultaneous retention of teacher+student logits.

9. Incorrect prefix masking if generated-token provenance is implicit.
- Mitigation: carry explicit token-aligned `generated_mask` through `MicroBatch` and use it for prefix suppression.

10. Unsupported multi-run semantics (per-run EMA and teacher context state explosion).
- Mitigation: explicitly reject `max_concurrent_runs > 1` for self-distill in v1.

---

## 11) First Test File (Implemented)

Created:

`tests/unit/train/rl/test_self_distill_core.py`

This file now contains CPU-only unit tests for core self-distillation math and EMA behavior, and is intended to run before full trainer/orchestrator wiring.

## 11.1 What this file should test first

1. Top-K + tail exactness on toy vocab (forward KL).
- Build tiny logits (`vocab <= 8`), compute full-vocab KL and Top-K+tail approximation.
- Assert equality (or near-equality) when Top-K includes all tokens.
- Assert expected approximation behavior for smaller K.

2. Top-K + tail exactness on toy vocab (reverse KL).
- Same pattern for reverse KL.

3. Symmetric divergence composition.
- Assert `sym == mix*fwd + (1-mix)*rev` numerically.

4. Student-selected Top-K invariant.
- Construct case where teacher top tokens differ.
- Assert chosen indices are from student distribution only.

5. Tail correction is active.
- Compare with a renormalized-topK-only baseline and assert outputs differ in a case with non-trivial tail mass.

6. Prefix masking behavior.
- Given generated-token mask with N valid tokens and `loss_mask_prefix_tokens = m`, assert first `m` generated positions contribute zero.

7. Prefix masking with sparse loss masks.
- Ensure masking counts generated positions correctly (not raw token positions).

8. Prefix masking edge case with overlong prefix.
- Assert that when `loss_mask_prefix_tokens` exceeds generated length, the resulting mask is all false.

9. EMA update rule correctness.
- Deterministic tensors: one update and multi-step update should match closed-form expectations.

## 11.2 Target API under test

The test file is written against pure-function APIs in trainer utilities:

- `compute_topk_tail_distill_loss(...)`
- `apply_prefix_generated_mask(...)`
- `ema_update_(teacher_params, student_params, alpha)`

These are expected to be CPU-testable with plain tensors and no distributed dependencies.

## 11.3 What is covered right now

The implemented tests currently cover:

- forward KL correctness when `top_k == vocab_size`;
- reverse KL correctness when `top_k == vocab_size`;
- symmetric composition from forward/reverse branches;
- student-selected Top-K invariant;
- explicit tail-correction behavior vs renormalized Top-K-only baseline;
- prefix-generated-token masking behavior (including sparse masks and overlong prefixes);
- EMA one-step and closed-form two-step updates.

## 11.4 Why this file first

This single file covers the highest-risk correctness surface:

- divergence math,
- Top-K/tail semantics,
- artifact prefix masking,
- EMA update behavior.

Once this file passes, integration wiring can proceed with much lower risk.

---

## 12) Acceptance Criteria

Self-distill mode is considered integrated when:

1. A config with `trainer.loss.type = "self_distill"` runs end-to-end with strict on-policy settings.
2. Trainer computes distillation loss using teacher-conditioned context and EMA teacher weights.
3. Prefix mask and Top-K tail correction are active and unit-tested.
4. EMA state is checkpointed/resumed correctly.
5. Existing non-self-distill modes remain unchanged.
