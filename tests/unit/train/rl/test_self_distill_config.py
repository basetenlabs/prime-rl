import pytest

from prime_rl.orchestrator.config import OrchestratorConfig, TeacherModelConfig
from prime_rl.rl import RLConfig
from prime_rl.trainer.rl.config import RLTrainerConfig, SelfDistillLossConfig


def test_self_distill_disables_fused_lm_head_auto_chunk():
    config = RLTrainerConfig(
        loss=SelfDistillLossConfig(),
        model={"fused_lm_head_chunk_size": "auto"},
    )
    assert config.model.fused_lm_head_chunk_size == "disabled"


def test_self_distill_rejects_multi_run():
    with pytest.raises(ValueError, match="max_concurrent_runs = 1"):
        RLTrainerConfig(loss=SelfDistillLossConfig(), max_concurrent_runs=2)


def test_rl_config_enforces_self_distill_on_policy():
    with pytest.raises(ValueError, match="strict on-policy execution"):
        RLConfig(
            trainer=RLTrainerConfig(loss=SelfDistillLossConfig(), max_async_level=1),
            orchestrator=OrchestratorConfig(max_async_level=1),
        )


def test_rl_config_sets_self_distill_flags_and_off_policy_guardrail():
    config = RLConfig(
        trainer=RLTrainerConfig(loss=SelfDistillLossConfig(), max_async_level=0),
        orchestrator=OrchestratorConfig(max_async_level=0, max_off_policy_steps=8),
    )
    assert config.orchestrator.self_distill is True
    assert config.orchestrator.max_off_policy_steps == 0


def test_rl_config_rejects_teacher_model_in_self_distill():
    with pytest.raises(ValueError, match="does not support orchestrator.teacher_model"):
        RLConfig(
            trainer=RLTrainerConfig(loss=SelfDistillLossConfig(), max_async_level=0),
            orchestrator=OrchestratorConfig(max_async_level=0, teacher_model=TeacherModelConfig()),
        )


def test_rl_config_custom_loss_does_not_require_teacher_tau():
    config = RLConfig(
        trainer=RLTrainerConfig(loss={"type": "custom", "import_path": "math.sin"}),
        orchestrator=OrchestratorConfig(),
    )
    assert config.trainer.loss.type == "custom"
