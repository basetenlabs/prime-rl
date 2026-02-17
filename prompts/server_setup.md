# Server Setup: Self-Distillation Replication on Baseten 8xH200

You are on a fresh Baseten training server (hostname like `baseten-training-job-*-multinode-0`). The goal is to set up and run a self-distillation training experiment replicating a paper.

## Step 1: Check the environment

Run these and report what you find:
```bash
nvidia-smi                    # Check GPUs (expecting 8xH200)
cat /etc/os-release           # OS info
which python3 && python3 --version
which uv || echo "uv not installed"
which git && git --version
df -h                         # Disk space
free -h                       # Memory
```

If `uv` is not installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

## Step 2: Clone the repo

```bash
cd /root  # or wherever makes sense
git clone https://github.com/basetenlabs/prime-rl.git
cd prime-rl
git checkout feature/self-distill-replication
```

This branch contains:
- Charlie's self-distillation implementation (from `feature/self-distill-mode`)
- Our additions: converted tool-use data, training config, conversion script, replication notes

## Step 3: Install dependencies

```bash
uv sync --all-extras
```

This will install all Python dependencies including PyTorch with CUDA, vLLM, verifiers, etc. It may take a while.

## Step 4: Verify GPU access from Python

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

Should show 8 H200 GPUs.

## Step 5: Create the custom verifiers environment

This is the main piece of work. We need a verifiers environment plugin that loads our tool-use dataset for the orchestrator.

Read the detailed spec at `prompts/create_tooluse_env.md` in the repo. Key points:

- The environment must provide a HuggingFace Dataset with columns: `example_id` (int), `prompt` (str), `task` (str), `messages` (list of dicts)
- Scaffold with `uv run prime env init tooluse-self-distill`, then customize
- Look at existing environments for reference: check the verifiers package source in `.venv` or the repo at https://github.com/PrimeIntellect-ai/verifiers/tree/ccef044/environments
- The `reverse_text` environment is the simplest reference
- For scoring: return reward=0 (self-distill loss ignores rewards)
- Data is at `replication_data/converted/train_data.json` (4,046 examples) and `replication_data/converted/eval_data.json` (68 examples)
- The converted data format:
```json
{"example_id": 0, "messages": [
    {"role": "user", "content": "Your task is to answer the user's question using available tools..."},
    {"role": "assistant", "content": "I need to use the sendHttpRequest tool..."}
]}
```
- `prompt` should be derived from `messages[0]["content"]` (the user message)
- `task` should be a constant string matching the env name

Install the environment once created:
```bash
uv run prime env install tooluse-self-distill --from-local ./environments/tooluse_self_distill
```
(Adjust the install command based on how you scaffold it)

## Step 6: Review and run the training config

The config is at `configs/self_distill_tooluse/train.toml`:

```toml
# Key settings matching the paper:
# - Model: Qwen/Qwen2.5-7B-Instruct
# - Loss: forward_kl (paper default), top_k=64, ema_alpha=0.01
# - Skip first 3 generated tokens from loss
# - LR: 2e-5, cosine schedule, 100 warmup steps
# - Batch size: 32, temperature: 1.0
# - 2 GPUs for inference, 6 for training
# - On-policy only (max_async_level=0)

[[orchestrator.env]]
id = "tooluse-self-distill"
name = "tooluse"
args = { dataset_path = "replication_data/converted/train_data.json", eval_dataset_path = "replication_data/converted/eval_data.json" }
```

You may need to adjust the env id/args based on how you implemented the environment in step 5.

Run training:
```bash
uv run trainer @ configs/self_distill_tooluse/train.toml
```

## Step 7: Monitor

Check wandb for `distill_kl` and `distill_tokens` metrics. The training should show decreasing KL divergence over time.

## What success looks like

After training, evaluate the model on the 68 eval examples. The paper reports:
- **Tool-use accuracy: ~70.6%** (regex match against golden API calls, order-insensitive)
- **Prior capability preservation: ~65.4 avg** across HellaSwag, HumanEval, IFEval, MMLU, TruthfulQA, Winogrande

The base Qwen2.5-7B-Instruct gets 42.9% on tool-use, so any significant improvement over that means the self-distillation is working.

## Important files in the repo

- `SELF_DISTILL_REPLICATION.md` — paper vs implementation comparison and replication plan
- `AGENTS.md` / `CLAUDE.md` — code style guidelines (read these first)
- `prompts/create_tooluse_env.md` — detailed spec for the verifiers environment
- `configs/self_distill_tooluse/train.toml` — training config
- `replication_data/converted/` — converted tool-use data
- `scripts/convert_tooluse_data.py` — data conversion script
- `src/prime_rl/trainer/rl/loss.py` — self-distill loss function
- `src/prime_rl/trainer/rl/ema.py` — EMA teacher utilities
- `src/prime_rl/trainer/rl/train.py` — training loop (self-distill branch)
- `src/prime_rl/orchestrator/self_distill_context.py` — teacher context construction
- `docs/on_policy_distillation.md` — self-distill mode documentation
