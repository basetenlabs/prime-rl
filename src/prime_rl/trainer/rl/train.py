from contextlib import nullcontext
import time
from datetime import timedelta

# Import environment before any other imports
# ruff: noqa: I001

from prime_rl.trainer.models.layers.attn import substitute_prime_rl_flash_attn
from prime_rl.trainer.rl.broadcast import setup_weight_broadcast
from prime_rl.utils.act_offloading import maybe_activation_offloading
import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.profiler import profile, ProfilerActivity, record_function
from loguru import logger
from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.trainer.multi_ckpt import setup_multi_checkpoint_manager
from prime_rl.trainer.optim import setup_optimizer, setup_multi_optimizer
from prime_rl.trainer.scheduler import setup_scheduler, setup_multi_scheduler
from prime_rl.trainer.rl.config import LossConfig, RLTrainerConfig, SelfDistillLossConfig
from prime_rl.trainer.rl.data import DataLoader, FakeDataLoader, TensorMicroBatch
from prime_rl.trainer.rl.ema import clone_params, copy_params_, ema_update_
from prime_rl.utils.cp import (
    setup_cp_params,
    shard_for_cp,
)
from prime_rl.utils.logger import setup_logger
from prime_rl.trainer.rl.loss import (
    apply_prefix_generated_mask,
    compute_entropy,
    compute_loss,
    compute_topk_tail_distill_loss,
    selective_log_softmax,
    setup_loss_fn,
    shift_logits,
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.trainer.model import (
    forward,
    setup_tokenizer,
    setup_model,
    is_tt_moe_model,
    get_load_balance_stats,
)
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.utils import (
    MemoryProfiler,
    Tensors,
    export_benchmark_json,
    get_ckpt_disk_metrics,
    setup_torch_distributed,
    print_benchmark,
    get_response_lengths,
)
from prime_rl.trainer.world import get_world
from prime_rl.trainer.runs import setup_multi_run_manager, Progress, get_multi_run_manager
from prime_rl.trainer.models.layers.lora import set_lora_num_tokens
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.metrics_server import HealthServer, MetricsServer, RunStats
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, resolve_latest_ckpt_step, to_col_format
from ring_flash_attn import substitute_hf_flash_attn
from torchtitan.distributed.utils import clip_grad_norm_


def _format_kl_heatmap(token_ids: torch.Tensor, kl_values: torch.Tensor, tokenizer) -> str:
    """Build an ANSI-colored string where each token's background reflects its KL divergence.

    Green = low KL (student agrees with teacher), Red = high KL (student disagrees).
    """
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    kl = kl_values.float().cpu()
    max_kl = max(kl.max().item(), 1e-6)

    # ANSI 256-color codes: 28=green, 64=olive, 100=dark yellow, 136=yellow, 172=orange, 208=red-orange, 196=red
    color_ramp = [28, 64, 100, 136, 172, 208, 196]
    parts = []
    for tok, kl_val in zip(tokens, kl):
        intensity = min(kl_val.item() / max_kl, 1.0)
        idx = min(int(intensity * (len(color_ramp) - 1)), len(color_ramp) - 1)
        color = color_ramp[idx]
        tok_str = tokenizer.convert_tokens_to_string([tok])
        parts.append(f"\033[48;5;{color}m\033[97m{tok_str}\033[0m")
    return "".join(parts)


def _compute_self_distill_loss_scale(
    micro_batches: list[TensorMicroBatch], loss_config: SelfDistillLossConfig
) -> int:
    total_distill_tokens = 0
    for micro_batch in micro_batches:
        loss_mask = micro_batch["loss_mask"]
        generated_mask = micro_batch["generated_mask"]
        effective_mask = apply_prefix_generated_mask(
            loss_mask=loss_mask,
            generated_mask=generated_mask,
            loss_mask_prefix_tokens=loss_config.loss_mask_prefix_tokens,
        )
        total_distill_tokens += int(effective_mask.sum().item())
    return max(total_distill_tokens, 1)


def _compute_self_distill_microbatch_loss(
    model: torch.nn.Module,
    config: RLTrainerConfig,
    loss_config: SelfDistillLossConfig,
    run_idx: int,
    step: int,
    micro_step: int,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    generated_mask: torch.Tensor,
    teacher_prompt_ids: list[list[int]] | None,
    loss_scale: int,
    ema_state: list[torch.Tensor],
    maybe_record_function,
    tokenizer=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    response_lengths = get_response_lengths(position_ids)
    split_input_ids = input_ids.squeeze(0).split(response_lengths)
    split_loss_mask = loss_mask.squeeze(0).split(response_lengths)
    split_generated_mask = generated_mask.squeeze(0).split(response_lengths)

    if teacher_prompt_ids is None:
        raise RuntimeError(
            f"Self-distill teacher prompts are missing at run_idx={run_idx}, step={step}, micro_step={micro_step}. "
            "Expected teacher_prompt_ids to be present in every micro batch."
        )
    if len(teacher_prompt_ids) != len(response_lengths):
        raise RuntimeError(
            f"Self-distill teacher prompt count mismatch at run_idx={run_idx}, step={step}, micro_step={micro_step}: "
            f"prompts={len(teacher_prompt_ids)} sequences={len(response_lengths)}"
        )

    effective_masks = [
        apply_prefix_generated_mask(
            loss_mask=seq_loss_mask.unsqueeze(0),
            generated_mask=seq_generated_mask.unsqueeze(0),
            loss_mask_prefix_tokens=loss_config.loss_mask_prefix_tokens,
        ).squeeze(0)
        for seq_loss_mask, seq_generated_mask in zip(split_loss_mask, split_generated_mask)
    ]

    teacher_selected_logits: list[torch.Tensor | None] = [None] * len(split_input_ids)

    # Pack all teacher sequences into a single forward call. The previous per-sequence
    # loop called forward() a data-dependent number of times (skipping masked sequences),
    # which caused FSDP ranks to issue different numbers of allgather collectives and deadlock.
    teacher_sequences: list[torch.Tensor] = []
    teacher_pos_ids_list: list[torch.Tensor] = []
    teacher_seq_lengths: list[int] = []
    teacher_prompt_lengths: list[int] = []
    included_seq_indices: list[int] = []

    for seq_idx, (seq_input_ids, seq_teacher_prompt_ids) in enumerate(
        zip(split_input_ids, teacher_prompt_ids)
    ):
        teacher_prompt = torch.tensor(seq_teacher_prompt_ids, dtype=torch.long, device=input_ids.device)
        teacher_input = torch.cat([teacher_prompt, seq_input_ids], dim=0)
        if teacher_input.numel() > config.model.seq_len:
            logger.warning(
                "Skipping self-distill teacher sequence that exceeds seq_len: "
                f"run_idx={run_idx}, step={step}, micro_step={micro_step}, sequence_index={seq_idx}, "
                f"teacher_prompt_tokens={teacher_prompt.numel()}, continuation_tokens={seq_input_ids.numel()}, "
                f"total_teacher_tokens={teacher_input.numel()}, model_seq_len={config.model.seq_len}"
            )
            continue
        teacher_sequences.append(teacher_input)
        teacher_pos_ids_list.append(torch.arange(teacher_input.numel(), device=input_ids.device, dtype=torch.long))
        teacher_seq_lengths.append(teacher_input.numel())
        teacher_prompt_lengths.append(teacher_prompt.numel())
        included_seq_indices.append(seq_idx)

    if not teacher_sequences:
        logger.warning(
            "All teacher sequences in micro batch exceeded seq_len, skipping self-distill loss: "
            f"run_idx={run_idx}, step={step}, micro_step={micro_step}"
        )
    else:
        packed_teacher_ids = torch.cat(teacher_sequences, dim=0).unsqueeze(0)
        packed_teacher_pos = torch.cat(teacher_pos_ids_list, dim=0).unsqueeze(0)

        model_params = list(model.parameters())
        student_param_snapshot = clone_params(model_params)
        copy_params_(model_params, ema_state)
        try:
            with torch.no_grad():
                with maybe_record_function("teacher_forward"), maybe_activation_offloading(config.model.ac_offloading):
                    teacher_out = forward(
                        model,
                        packed_teacher_ids,
                        packed_teacher_pos,
                        labels=None,
                        temperature=None,
                    )
                teacher_logits = teacher_out.get("logits")
                if teacher_logits is None:
                    raise RuntimeError(
                        "Teacher forward did not return logits in self-distill mode. "
                        "Ensure fused LM head is disabled."
                    )

                teacher_aligned_logits = shift_logits(teacher_logits).squeeze(0)
                split_teacher_logits = teacher_aligned_logits.split(teacher_seq_lengths, dim=0)

                for teacher_idx, (seq_teacher_logits, prompt_len) in enumerate(
                    zip(split_teacher_logits, teacher_prompt_lengths)
                ):
                    orig_seq_idx = included_seq_indices[teacher_idx]
                    seq_input_ids = split_input_ids[orig_seq_idx]
                    seq_effective_mask = effective_masks[orig_seq_idx]
                    if not seq_effective_mask.any():
                        continue
                    continuation_logits = seq_teacher_logits[prompt_len:]
                    if continuation_logits.shape[0] != seq_input_ids.numel():
                        raise RuntimeError(
                            "Failed to align teacher logits to continuation tokens: "
                            f"run_idx={run_idx}, step={step}, micro_step={micro_step}, sequence_index={orig_seq_idx}, "
                            f"continuation_tokens={seq_input_ids.numel()}, aligned_teacher_tokens={continuation_logits.shape[0]}"
                        )
                    teacher_selected_logits[orig_seq_idx] = continuation_logits[seq_effective_mask].detach()
        finally:
            copy_params_(model_params, student_param_snapshot)

    with maybe_record_function("forward"), maybe_activation_offloading(config.model.ac_offloading):
        student_out = forward(
            model,
            input_ids,
            position_ids,
            labels=None,
            temperature=None,
        )
    student_logits = student_out.get("logits")
    if student_logits is None:
        raise RuntimeError("Self-distill mode requires full logits; set model.fused_lm_head_chunk_size='disabled'.")

    student_aligned_logits = shift_logits(student_logits)
    student_logprobs = selective_log_softmax(student_aligned_logits, input_ids)
    student_entropy = compute_entropy(student_aligned_logits)

    split_student_logits = student_aligned_logits.squeeze(0).split(response_lengths, dim=0)

    distill_token_losses: list[torch.Tensor] = []
    heatmap_seq: tuple[torch.Tensor, torch.Tensor] | None = None
    total_distill_loss = torch.zeros((), device=input_ids.device, dtype=torch.float32)
    for seq_idx, (student_seq_logits, seq_effective_mask, teacher_seq_logits) in enumerate(
        zip(split_student_logits, effective_masks, teacher_selected_logits)
    ):
        if teacher_seq_logits is None:
            continue
        student_selected_logits = student_seq_logits[seq_effective_mask]
        if student_selected_logits.shape[0] != teacher_seq_logits.shape[0]:
            raise RuntimeError(
                "Teacher/student selected token count mismatch in self-distill mode: "
                f"run_idx={run_idx}, step={step}, micro_step={micro_step}, sequence_index={seq_idx}, "
                f"student_tokens={student_selected_logits.shape[0]}, teacher_tokens={teacher_seq_logits.shape[0]}"
            )

        seq_token_losses = compute_topk_tail_distill_loss(
            student_logits=student_selected_logits.unsqueeze(0),
            teacher_logits=teacher_seq_logits.unsqueeze(0),
            top_k=loss_config.top_k,
            divergence=loss_config.divergence,
            symmetric_mix=loss_config.symmetric_mix,
        ).squeeze(0)
        distill_token_losses.append(seq_token_losses)
        total_distill_loss = total_distill_loss + seq_token_losses.sum()

        if heatmap_seq is None:
            masked_token_ids = split_input_ids[seq_idx][seq_effective_mask]
            heatmap_seq = (masked_token_ids, seq_token_losses)

    if micro_step == 0 and heatmap_seq is not None and tokenizer is not None:
        hm_ids, hm_kl = heatmap_seq
        heatmap_str = _format_kl_heatmap(hm_ids, hm_kl, tokenizer)
        logger.info(f"KL HEATMAP (step {step}):\n{heatmap_str}")

    if distill_token_losses:
        distill_kl = torch.cat(distill_token_losses, dim=0)
    else:
        distill_kl = torch.zeros(0, device=input_ids.device, dtype=torch.float32)
        # When no tokens survive the effective mask, total_distill_loss is a plain zero with no
        # grad_fn. Backward on a detached scalar crashes, and — worse — the missing FSDP allgather
        # collectives deadlock every other rank. Multiplying by 0.0 through the student logits
        # keeps the tensor on the computation graph so backward issues the same collectives as
        # every other rank while contributing zero gradient.
        total_distill_loss = total_distill_loss + 0.0 * student_aligned_logits.sum()
    distill_tokens = torch.tensor([float(distill_kl.numel())], device=input_ids.device, dtype=torch.float32)

    scaled_loss = total_distill_loss / loss_scale
    loss_metrics = {
        "distill_kl": distill_kl,
        "distill_tokens": distill_tokens,
    }
    return scaled_loss, loss_metrics, student_logprobs, student_entropy


@clean_exit
@logger.catch(reraise=True)
def train(config: RLTrainerConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "trainer" / f"rank_{world.rank}.log" if config.log.file else None,
        json_logging=config.log.json_logging,
    )
    logger.info(f"Starting RL trainer in {world} in {config.output_dir}")

    # Print warning if running in benchmark mode
    if config.bench is not None:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Setup the monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)

    # Setup heartbeat (only on rank 0)
    heart = None
    if config.heartbeat is not None and world.is_master:
        logger.info("Initializing heartbeat")
        heart = Heartbeat(config.heartbeat.url)

    # Setup metrics server (full on master, health-only on other nodes' local rank 0)
    metrics_server = None
    health_server = None
    if config.metrics_server is not None and world.local_rank == 0:
        if world.is_master:
            logger.info(f"Initializing metrics server on port {config.metrics_server.port}")
            metrics_server = MetricsServer(config.metrics_server)
            metrics_server.start()
        else:
            logger.info(f"Initializing health server on port {config.metrics_server.port}")
            health_server = HealthServer(config.metrics_server.port, config.metrics_server.host)
            health_server.start()

    # Set precision
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds), enable_gloo=config.model.fsdp_cpu_offload
    )
    torch.set_float32_matmul_precision("high")

    # Setup multi run manager and offsets (including LoRA validation/scaling hooks if applicable)
    multi_run_manager = setup_multi_run_manager(
        config.output_dir, config.max_concurrent_runs, torch.device("cuda", world.local_rank), config.model.lora
    )

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model)

    # For single-run, check for checkpoint to resume from
    checkpoint_step = None
    if config.max_concurrent_runs == 1:
        # Set up checkpoint manager for single-run
        logger.info(f"Initializing checkpoint managers ({config.ckpt})")
        ckpt_manager, weight_ckpt_manager = setup_ckpt_managers(config.output_dir, config.ckpt, config.model.lora)

        if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
            if config.ckpt.resume_step == -1:
                checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
            else:
                checkpoint_step = config.ckpt.resume_step
    else:
        # Multi-run uses per-run checkpointing via MultiCheckpointManager
        ckpt_manager, weight_ckpt_manager = setup_multi_checkpoint_manager(config.output_dir)
        logger.info("Initialized multi-run checkpoint manager")

    # Initialize the model and tokenizer
    logger.info(f"Initializing model ({config.model})")
    loading_from_ckpt_later = config.ckpt and checkpoint_step is not None
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)

    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)

    # Set up the loss function
    logger.info(f"Setting up loss function ({config.loss})")
    is_self_distill = isinstance(config.loss, SelfDistillLossConfig)
    if is_self_distill:
        loss_fn = None
    else:
        loss_fn = setup_loss_fn(config.loss)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")

    if config.max_concurrent_runs == 1:
        optimizer = setup_optimizer(
            config.optim,
            list(model.named_parameters()),
            parallel_dims,
            lora=config.model.lora is not None,
            cpu_offload=config.model.optim_cpu_offload,
        )
        scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
    else:
        optimizer = setup_multi_optimizer(config.optim, parallel_dims)
        scheduler = setup_multi_scheduler(optimizer, config.scheduler, config.max_steps)

        # Register checkpoint loading callback at index 1 (after scheduler creation at index 0)
        def load_run_checkpoint(_optimizer, idx: int) -> None:
            ckpt_manager.load_run(idx, optimizer, scheduler)

        optimizer.register_post_creation_callback(load_run_checkpoint, index=1)

    logger.info(f"Using `{config.scheduler.type}` scheduler ({config.scheduler})")

    # Set up weight broadcast
    logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
    weight_broadcast = setup_weight_broadcast(config.output_dir, config.weight_broadcast, config.model.lora)

    if parallel_dims.cp_enabled:
        substitute_hf_flash_attn(parallel_dims.world_mesh["cp"].get_group(), heads_k_stride=1)
        substitute_prime_rl_flash_attn(
            parallel_dims.world_mesh["cp"].get_group(),
            heads_k_stride=1,
            attn_impl=config.model.attn,
        )

    # Optionally, resume training from a checkpoint
    progress = Progress()
    if checkpoint_step is not None:
        ckpt_manager.load(checkpoint_step, model, [optimizer], scheduler, progress)
        logger.info(f"Resuming training from checkpoint step {checkpoint_step}")

    ema_state: list[torch.Tensor] | None = None
    if is_self_distill:
        model_params = list(model.parameters())
        ema_state = clone_params(model_params)
        if checkpoint_step is not None and ckpt_manager is not None:
            ema_from_checkpoint = ckpt_manager.load_ema(checkpoint_step, map_location=model_params[0].device)
            if ema_from_checkpoint is None:
                logger.warning(
                    f"No EMA checkpoint state found at step {checkpoint_step}; initializing EMA teacher from student weights."
                )
            else:
                if len(ema_from_checkpoint) != len(model_params):
                    raise RuntimeError(
                        f"EMA checkpoint parameter count mismatch: ckpt={len(ema_from_checkpoint)} model={len(model_params)}"
                    )
                ema_state = []
                for model_param, ema_param in zip(model_params, ema_from_checkpoint):
                    if model_param.shape != ema_param.shape:
                        raise RuntimeError(
                            f"EMA checkpoint parameter shape mismatch: ckpt={ema_param.shape} model={model_param.shape}"
                        )
                    ema_state.append(ema_param.to(device=model_param.device, dtype=model_param.dtype))
                logger.info(f"Loaded EMA checkpoint state from step {checkpoint_step}")

    logger.info(
        f"Starting from step {progress.step} (total_tokens={progress.total_tokens}, total_samples={progress.total_samples})"
    )

    # Set up the data loader (Optionally, use a fake data loader for debugging)
    logger.info(f"Initializing data loader ({config.data})")
    if config.data.fake:
        dataloader = FakeDataLoader(config.data.fake, config.model.seq_len, parallel_dims.world_mesh["dp"].size())
    else:
        dataloader = DataLoader(
            config.output_dir,
            progress.step,
            parallel_dims.world_mesh["dp"].size(),
            config.model.seq_len,
            config.model.cp,
            tokenizer,
            config.rollout_transport,
        )

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    is_first_step = True
    maybe_record_function = nullcontext
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        maybe_record_function = record_function
    while True:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        is_last_step = config.max_steps is not None and progress.step == config.max_steps

        # Broadcast weights at every step, (except step 0, because no need to broadcast the base model)
        # Also, with NCCL broadcast, we do not broadcast weights the last async level step as the orchestrator is already finished and will not initialize the receive on the inference; for filesystem broadcast, we do "broadcast" until the final step to allow to resume from the broadcast directory
        last_async_level_steps = config.max_steps and progress.step >= config.max_steps - config.max_async_level
        if progress.step > 0 and (not last_async_level_steps or config.weight_broadcast.type == "filesystem"):
            broadcast_weights_start_time = time.perf_counter()
            weight_broadcast.broadcast_weights(model, step=progress.step)
            broadcast_weights_time = time.perf_counter() - broadcast_weights_start_time
            # Clean up old broadcast directories (unless at ckpt interval if using filesystem weight broadcast)
            ckpt_interval = config.ckpt and config.ckpt.interval
            interval_to_keep = ckpt_interval if config.weight_broadcast.type == "filesystem" else None
            if config.weight_broadcast.type == "filesystem":
                weight_broadcast.maybe_clean(config.max_async_level, interval_to_keep)
        else:
            broadcast_weights_time = 0
            # Usually the broadcast will set this. If broadcast is skipped, we need to reset this here.
            for idx in multi_run_manager.used_idxs:
                multi_run_manager.ready_to_update[idx] = False

        if (
            ckpt_manager is not None
            and (config.ckpt and config.ckpt.interval)
            and not (is_first_step or is_last_step)
            and progress.step % config.ckpt.interval == 0
        ):
            # Single-run: Save full checkpoint
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
            if is_self_distill and ema_state is not None:
                ckpt_manager.save_ema(progress.step, ema_state)
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time

            # Maybe clean up old checkpoints
            ckpt_manager.maybe_clean()

            # Save weight checkpoint
            if weight_ckpt_manager is not None:
                logger.info(f"Saving weight checkpoint at step {progress.step}")
                weight_ckpt_manager.save(progress.step, model, tokenizer)

                # Maybe clean up old weight checkpoint
                weight_ckpt_manager.maybe_clean()
        elif config.max_concurrent_runs > 1:
            # Multi-run: Save per-run checkpoints (each run has its own interval from orchestrator config)
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(optimizer, scheduler)
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time
            ckpt_manager.maybe_clean()
        else:
            save_ckpt_time = 0

        # Break if we have reached the maximum number of steps
        if config.max_steps is not None and progress.step >= config.max_steps:
            break

        logger.debug(f"Starting training step {progress.step}")
        step_start_time = time.perf_counter()

        # Wait for the batch to be available
        logger.debug("Waiting for training batch to arrive")
        wait_for_batch_start_time = time.perf_counter()
        dataloader.wait_for_batch()
        wait_for_batch_time = time.perf_counter() - wait_for_batch_start_time
        logger.debug(f"Waited for batch to arrive for {wait_for_batch_time:.2f} seconds")

        # Load the training batch
        logger.debug("Loading batch")
        load_data_start_time = time.perf_counter()
        micro_batches = dataloader.get_batch()
        load_data_time = time.perf_counter() - load_data_start_time
        logger.debug(f"Loaded batch in {load_data_time:.2f} seconds")

        batch_size = len(micro_batches)
        memory_profiler = None
        if config.memory_profiler_path is not None:
            memory_profiler = MemoryProfiler(progress.step, config.memory_profiler_path)

        forward_backward_start_time = time.perf_counter()
        seq_len = micro_batches[0]["input_ids"].shape[1]

        # Normalize by the local number of unmasked tokens in the batch (per-batch length normalization)
        if is_self_distill:
            assert isinstance(config.loss, SelfDistillLossConfig)
            loss_scale = _compute_self_distill_loss_scale(micro_batches, config.loss)
        elif isinstance(config.loss, LossConfig) and config.loss.ratio_type == "token":
            loss_scale = sum(micro_batch["loss_mask"].sum().item() for micro_batch in micro_batches)
        else:
            loss_scale = batch_size
        loss_scale = max(loss_scale, 1)

        logger.debug(f"Starting forward and backward pass ({batch_size=})")
        tensors = Tensors()  # Used to accumulate tensor statistics across micro-batches and ranks for logging
        cp_enabled = parallel_dims.cp_enabled
        cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if cp_enabled else 0
        cp_group = parallel_dims.world_mesh["cp"].get_group() if cp_enabled else None
        cp_size = parallel_dims.cp
        if is_self_distill and cp_enabled:
            raise NotImplementedError("Self-distill mode does not support context parallelism in v1.")

        for micro_step, micro_batch in enumerate(micro_batches):
            input_ids = micro_batch["input_ids"].to("cuda")
            position_ids = micro_batch["position_ids"].to("cuda")
            advantages = micro_batch["advantages"].to("cuda")
            loss_mask = micro_batch["loss_mask"].to("cuda")
            generated_mask = micro_batch["generated_mask"].to("cuda")
            inference_logprobs = micro_batch["inference_logprobs"].to("cuda")
            teacher_logprobs = (
                micro_batch["teacher_logprobs"].to("cuda") if micro_batch["teacher_logprobs"] is not None else None
            )
            teacher_prompt_ids = micro_batch.get("teacher_prompt_ids")
            run_token_counts = micro_batch["lora_num_tokens"]
            nonzero_run_idxs = torch.nonzero(run_token_counts > 0, as_tuple=False).flatten()
            run_idx = int(nonzero_run_idxs[0].item()) if nonzero_run_idxs.numel() > 0 else -1

            # Multimodal fields (Qwen3-VL) - only present for VLM training
            pixel_values = (
                micro_batch["pixel_values"].to("cuda") if micro_batch.get("pixel_values") is not None else None
            )
            image_grid_thw = (
                micro_batch["image_grid_thw"].to("cuda") if micro_batch.get("image_grid_thw") is not None else None
            )
            if is_self_distill and pixel_values is not None:
                raise NotImplementedError("Self-distill mode does not support VLM/multimodal training in v1.")

            labels = shift_tensor_left(input_ids)

            # VLM + CP is not supported: MRoPE requires global positions but CP shards the sequence
            if cp_enabled and pixel_values is not None:
                raise NotImplementedError("Context parallelism is not supported with VLM/multimodal training")

            if cp_enabled:
                input_ids, forward_position_ids = setup_cp_params(input_ids, position_ids, cp_rank, cp_size, cp_group)
                labels = shard_for_cp(labels, cp_rank=cp_rank, cp_world_size=cp_size)
            else:
                forward_position_ids = position_ids

            if config.model.lora:
                lora_num_tokens = micro_batch["lora_num_tokens"].to("cuda")
                if cp_enabled:
                    chunk_size = input_ids.shape[1]  # We pad to multiple of cp so this should be fine
                    logger.debug(f"[Rank {world.rank}] {cp_rank=} {cp_size=} {cp_group=} {chunk_size=}")
                    # Convert to cumsum, adjust for CP chunk, convert back to num_tokens
                    cu_offsets = lora_num_tokens.cumsum(dim=0, dtype=torch.int32)
                    adjusted_cu = torch.clip(cu_offsets - chunk_size * cp_rank, min=0, max=chunk_size)
                    lora_num_tokens = torch.diff(
                        adjusted_cu, prepend=torch.tensor([0], device=adjusted_cu.device, dtype=adjusted_cu.dtype)
                    )
                set_lora_num_tokens(lora_num_tokens)

            temperatures = micro_batch["temperatures"].to("cuda")

            # Shard temperatures for context parallelism if enabled
            if cp_enabled:
                temperatures = shard_for_cp(temperatures, cp_rank=cp_rank, cp_world_size=cp_size)

            if is_self_distill:
                assert isinstance(config.loss, SelfDistillLossConfig)
                assert ema_state is not None
                loss, loss_tensors, trainer_logprobs_for_logging, entropy_for_logging = _compute_self_distill_microbatch_loss(
                    model=model,
                    config=config,
                    loss_config=config.loss,
                    run_idx=run_idx,
                    step=progress.step,
                    micro_step=micro_step,
                    input_ids=input_ids,
                    position_ids=forward_position_ids,
                    loss_mask=loss_mask,
                    generated_mask=generated_mask,
                    teacher_prompt_ids=teacher_prompt_ids,
                    loss_scale=loss_scale,
                    ema_state=ema_state,
                    maybe_record_function=maybe_record_function,
                    tokenizer=tokenizer,
                )
            else:
                # Forward pass with per-token temperatures
                with maybe_record_function("forward"), maybe_activation_offloading(config.model.ac_offloading):
                    out = forward(
                        model,
                        input_ids,
                        forward_position_ids,
                        labels=labels,
                        temperature=temperatures,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                    )

                if out.get("logprobs") is None:
                    # VanillaOutputLinear was used - need to compute logprobs externally with per-token temps
                    assert out.get("logits") is not None, "Logits must be provided to compute logprobs"
                    logits = out["logits"]
                    # Per-token temperature scaling: temperatures is [batch, seq], logits is [batch, seq, vocab]
                    scaled_logits = logits / temperatures.unsqueeze(-1)
                    out["logprobs"] = selective_log_softmax(scaled_logits, labels)
                    out["entropy"] = compute_entropy(scaled_logits)
                # else: FusedOutputLinear was used - logprobs already computed with per-token temperatures

                if cp_enabled:
                    logprobs = dist_nn.all_gather(out["logprobs"], group=cp_group)
                    out["logprobs"] = torch.cat(logprobs, dim=1)

                    entropies = [torch.zeros_like(out["entropy"]) for _ in range(cp_size)]
                    dist.all_gather(entropies, out["entropy"], group=cp_group)
                    out["entropy"] = torch.cat(entropies, dim=1)

                vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
                # This is not really necessary as the first token should be masked out, but we do it anyway to be sure
                out["logprobs"] = shift_tensor_right(
                    out["logprobs"], pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item()
                )
                out["entropy"] = shift_tensor_right(
                    out["entropy"], pad_value=torch.log(torch.tensor(float(vocab_size))).item()
                )

                # Compute loss
                response_lengths = get_response_lengths(position_ids)
                assert loss_fn is not None
                loss, loss_tensors = compute_loss(
                    trainer_logprobs=out["logprobs"].squeeze().split(response_lengths),
                    inference_logprobs=inference_logprobs.squeeze().split(response_lengths),
                    teacher_logprobs=teacher_logprobs.squeeze().split(response_lengths)
                    if teacher_logprobs is not None
                    else None,
                    advantages=advantages.squeeze().split(response_lengths),
                    loss_mask=loss_mask.squeeze().split(response_lengths),
                    loss_fn=loss_fn,
                    loss_scale=loss_scale,
                )
                trainer_logprobs_for_logging = out["logprobs"]
                entropy_for_logging = out["entropy"]

            # Backward pass
            with maybe_record_function("backward"):
                loss.backward()

            # Add relevant tensors to tensor dict for logging purposes
            tensors["trainer_probs"].append(torch.exp(trainer_logprobs_for_logging)[loss_mask].detach().to("cpu"))
            tensors["inference_probs"].append(torch.exp(inference_logprobs)[loss_mask].detach().to("cpu"))
            tensors["entropy"].append(entropy_for_logging[loss_mask].detach().to("cpu"))
            tensors["loss"].append(loss.detach().to("cpu").unsqueeze(0))

            if is_tt_moe_model(model):
                load_balance_stats = get_load_balance_stats(model)
                for k, v in load_balance_stats.items():
                    if v is not None:
                        tensors[k].append(v)

            # Add loss tensors to tensor dict for logging purposes
            for key, loss_tensor in loss_tensors.items():
                loss_tensor = loss_tensor.detach().to("cpu")
                tensors[key].append(loss_tensor)

            # Debug log with *local, micro step* stats
            micro_step_message = f"Micro Step {micro_step}/{len(micro_batches)} | Loss: {tensors['loss'][-1].mean().item():.4f} | Entropy: {tensors['entropy'][-1].mean().item():.4f}"
            if "mismatch_kl" in tensors:
                micro_step_message += f" | Mismatch KL: {tensors['mismatch_kl'][-1].mean().item():.4f}"
            if "max_vio" in tensors:
                micro_step_message += f" | Max Vio: {tensors['max_vio'][-1].mean().item():.4f}"
            logger.debug(micro_step_message)

        # Optionally, clip the gradients

        grad_norm = clip_grad_norm_(
            model.parameters(), max_norm=config.optim.max_norm, ep_enabled=parallel_dims.ep_enabled
        )
        if grad_norm.device.type == "cpu":
            grad_norm = grad_norm.to(torch.device("cuda"))

        # Update the model parameters
        optimizer.step()
        if is_self_distill:
            assert isinstance(config.loss, SelfDistillLossConfig)
            assert ema_state is not None
            ema_update_(teacher_params=ema_state, student_params=model.parameters(), alpha=config.loss.ema_alpha)
        optimizer.zero_grad()

        # Update learning rate scheduler
        scheduler.step()

        if config.max_concurrent_runs == 1:
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer.get_current_lr()
        forward_backward_time = time.perf_counter() - forward_backward_start_time

        # Optionally, dump memory snapshot
        if memory_profiler is not None:
            memory_profiler.step()

        # Synchronize the tensor metrics across all steps and ranks
        tensor_stats = tensors.compute_stats()

        # Compute step metrics
        num_local_tokens = seq_len * batch_size
        num_tokens = parallel_dims.world_mesh["dp"].size() * num_local_tokens
        progress.total_tokens += num_tokens
        progress.total_samples += batch_size
        perf_counter = get_perf_counter(model, seq_len)
        perf_counter.count_tokens(num_tokens)
        throughput = perf_counter.get_tokens_per_second() or 0
        mfu = perf_counter.get_mfu() or 0
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3  # GiB

        # Log step metrics
        step_time = time.perf_counter() - step_start_time
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Loss: {tensor_stats['loss/mean']:.4f} | Entropy: {tensor_stats['entropy/mean']:.4f}"
        if "mismatch_kl/mean" in tensor_stats:
            step_message += f" | Mismatch KL: {tensor_stats['mismatch_kl/mean']:.4f}"
        step_message += f" | Grad. Norm: {grad_norm:.4f} | LR: {current_lr:.2e} | Throughput: {throughput:.0f} tokens/s | MFU: {mfu:.1f}% | Peak Mem.: {peak_memory:.1f} GiB"
        if "max_vio/mean" in tensor_stats:
            step_message += f" | Max Vio: {tensor_stats['max_vio/mean']:.4f}"
        logger.success(step_message)

        # Log performance metrics
        perf_metrics = {
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/mfu": mfu,
            "perf/peak_memory": peak_memory,
            "step": progress.step,
        }
        monitor.log(perf_metrics, step=progress.step)

        # Log optimizer metrics
        optim_metrics = {
            "optim/lr": current_lr,
            "optim/grad_norm": grad_norm.item(),
            "step": progress.step,
        }
        monitor.log(optim_metrics, step=progress.step)

        # Log tensor stats
        tensor_stats["step"] = progress.step
        monitor.log(tensor_stats, step=progress.step)

        # Log time metrics
        time_metrics = {
            "time/step": step_time,
            "time/wait_for_batch": wait_for_batch_time,
            "time/load_data": load_data_time,
            "time/broadcast_weights": broadcast_weights_time,
            "time/save_ckpt": save_ckpt_time,
            "time/forward_backward": forward_backward_time,
            "step": progress.step,
        }
        monitor.log(time_metrics, step=progress.step)

        # Log disk metrics
        disk_metrics = get_ckpt_disk_metrics(config.output_dir)
        disk_metrics["step"] = progress.step
        monitor.log(disk_metrics, step=progress.step)

        # Update Prometheus metrics if configured
        if metrics_server is not None:
            metrics_server.update(
                step=progress.step,
                loss=tensor_stats["loss/mean"],
                throughput=throughput,
                grad_norm=grad_norm.item(),
                peak_memory_gib=peak_memory,
                learning_rate=current_lr,
                mfu=mfu,
                entropy=tensor_stats.get("entropy/mean", 0.0),
                mismatch_kl=tensor_stats.get("mismatch_kl/mean", 0.0),
            )
            # Update run/LoRA metrics
            multi_run_manager = get_multi_run_manager()
            runs_discovered = len(list(config.output_dir.glob("run_*")))
            run_stats = []
            for idx in multi_run_manager.used_idxs:
                run_id = multi_run_manager.idx_2_id[idx]
                run_progress = multi_run_manager.progress[idx]
                if config.max_concurrent_runs == 1:
                    lr = optimizer.param_groups[0]["lr"]
                else:
                    lr = optimizer.get_current_lr(idx) if optimizer.optimizers[idx] else 0.0
                run_stats.append(
                    RunStats(
                        run_id=run_id,
                        step=run_progress.step,
                        total_tokens=run_progress.total_tokens,
                        learning_rate=lr,
                        ready=multi_run_manager.ready_to_update[idx],
                    )
                )
            metrics_server.update_runs(
                runs_discovered=runs_discovered,
                runs_max=multi_run_manager.max_runs,
                run_stats=run_stats,
            )

        progress.step += 1
        is_first_step = False

        # Send heartbeat if configured
        if heart is not None:
            heart.beat()

    if config.trace_path:
        prof.__exit__(None, None, None)
        config.trace_path.mkdir(parents=True, exist_ok=True)
        trace_file = str(config.trace_path / f"trace_{dist.get_rank()}.json.gz")
        logger.info(f"Saving trace to {trace_file}")
        prof.export_chrome_trace(trace_file)
        logger.info(f"Saved trace to {trace_file}")

    # Write final checkpoint (only for single-run mode; multi-run checkpoints are managed by MultiCheckpointManager)
    if config.max_concurrent_runs == 1 and ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
        if is_self_distill and ema_state is not None:
            ckpt_manager.save_ema(progress.step, ema_state)
        ckpt_manager.maybe_clean()

    if config.max_concurrent_runs == 1 and weight_ckpt_manager is not None:
        logger.info("Writing final weight checkpoint")
        weight_ckpt_manager.save(progress.step, model, tokenizer)
        weight_ckpt_manager.maybe_clean()

    logger.info(f"Peak memory: {max(to_col_format(monitor.history)['perf/peak_memory']):.1f} GiB")
    logger.success("RL trainer finished!")

    # Stop metrics/health server if configured
    if metrics_server is not None:
        metrics_server.stop()
    if health_server is not None:
        health_server.stop()

    # Optionally, print benchmark table and export JSON
    if config.bench is not None and world.is_master:
        history = to_col_format(monitor.history)
        print_benchmark(history)
        if config.bench.output_json:
            export_benchmark_json(history, config.bench.output_json)
            logger.info(f"Benchmark results written to {config.bench.output_json}")


def main():
    """Main entry-point for RL trainer. Run using `uv run trainer`"""

    train(parse_argv(RLTrainerConfig))


if __name__ == "__main__":
    main()
