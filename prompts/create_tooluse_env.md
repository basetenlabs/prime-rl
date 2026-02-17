# Task: Create a custom verifiers environment for tool-use self-distillation

## Goal

Create a minimal `verifiers` environment plugin that loads a tool-use dataset (ToolAlpaca) and provides it to prime-rl's orchestrator for self-distillation training. The environment lives in the `verifiers` ecosystem (https://github.com/PrimeIntellect-ai/verifiers) and gets installed via `prime env install`.

## Background

We're replicating the paper "Self-Distillation Enables Continual Learning" (arXiv:2601.19897) using prime-rl's self-distillation mode (on branch `feature/self-distill-replication` in `basetenlabs/prime-rl`). The paper's original code is at https://github.com/idanshen/Self-Distillation.

In prime-rl, the orchestrator uses `verifiers` environments to:
1. **Provide a dataset** — a HuggingFace Dataset with required columns
2. **Run rollouts** — send prompts to the model for generation
3. **Score responses** — return a reward

For self-distillation, rewards don't affect training (the loss is pure KL divergence between student and EMA teacher). So the scoring can return reward=0 or optionally do regex matching for monitoring.

## What the buffer expects

The orchestrator's buffer (`src/prime_rl/orchestrator/buffer.py`) requires these columns in the dataset:
```python
assert "example_id" in self.dataset.column_names  # int, unique
assert "prompt" in self.dataset.column_names        # the chat prompt for generation
assert "task" in self.dataset.column_names           # environment name string
```

Additionally, self-distillation mode (`src/prime_rl/orchestrator/self_distill_context.py`) reads `messages` from each example to build the teacher's demonstration-conditioned prompt:
```python
example = example_lookup[example_id]
messages = example.get("messages")  # OpenAI-style [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
```

So the dataset must have columns: `example_id`, `prompt`, `task`, `messages`.

## The converted data

We already have converted data at `replication_data/converted/train_data.json` with this format:
```json
{
    "example_id": 0,
    "messages": [
        {"role": "user", "content": "Your task is to answer the user's question using available tools..."},
        {"role": "assistant", "content": "I need to use the sendHttpRequest tool..."}
    ]
}
```

There are 4,046 training examples and 68 eval examples.

The `prompt` field should be derived from `messages` — specifically the user message content, formatted as a chat prompt that gets sent to the inference server for generation.

The `task` field should be a constant string matching the environment name (e.g., `"tooluse"`).

## How existing environments work

Here's how environments are referenced in prime-rl configs:
```toml
[[orchestrator.env]]
id = "primeintellect/math-env"
name = "hendrycks-math"
args = { dataset_name = "PrimeIntellect/Hendrycks-Math", dataset_subset = "default" }
```

The orchestrator loads them via:
```python
import verifiers as vf
envs = [vf.load_environment(env_id, **env.args) for env_id, env in zip(env_ids, config.env)]
train_env_group = vf.EnvGroup(envs=envs, env_names=[...])
train_dataset = train_env_group.get_dataset(seed=config.buffer.seed)
```

You can scaffold a new environment with `prime env init my-env`. Check the verifiers repo for examples of simple environments: https://github.com/PrimeIntellect-ai/verifiers/tree/ccef044/environments

The `reverse_text` environment is the simplest reference — look at it for the minimum structure.

## What you need to build

A verifiers environment plugin called something like `tooluse-self-distill` that:

1. Takes an arg like `dataset_path` pointing to a local JSON file (or `dataset_name` for a HF dataset)
2. Loads the data and returns a HuggingFace Dataset with columns: `example_id`, `prompt`, `task`, `messages`
3. The `prompt` column should be the chat-formatted user message (from `messages[0]["content"]`)
4. The `task` column should be a constant string (the environment name)
5. For scoring: return reward=0 (since self-distill doesn't use rewards), OR optionally do regex matching against golden answers for monitoring
6. For eval data, the paper uses regex matching against ground-truth API calls accounting for argument order variations

## Paper's scoring approach (for reference)

From the paper's `main.py`, evaluation uses regex matching against `golden_answer`:
```python
# golden_answer is a list like:
# [{"Action": "sendHttpRequest", "Action_Input": "{\"method\": \"POST\", ...}"}]
# Accuracy was evaluated using regex matching against the ground-truth API call,
# accounting for variations in argument ordering.
```

The eval data (`replication_data/converted/eval_data.json`) has:
```json
{
    "example_id": 0,
    "prompt": "Your task is to answer...",
    "instruction": "Can you help me send a POST request...",
    "golden_answer": [{"Action": "sendHttpRequest", "Action_Input": "..."}]
}
```

## Config integration

Once the environment exists, the training config at `configs/self_distill_tooluse/train.toml` should use:
```toml
[[orchestrator.env]]
id = "tooluse-self-distill"
name = "tooluse"
args = { dataset_path = "replication_data/converted/train_data.json" }
```

## Key constraints

- Follow the patterns from existing verifiers environments
- The environment must work with the self-distillation orchestrator flow
- Keep it minimal — we primarily need dataset loading and basic rollout support
- Read `AGENTS.md` and `CLAUDE.md` in the repo root for code style guidelines
