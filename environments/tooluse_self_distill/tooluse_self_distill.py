import json
import re
from pathlib import Path

from datasets import Dataset

import verifiers as vf


def load_environment(
    dataset_path: str = "replication_data/converted/train_data.json",
    eval_dataset_path: str | None = None,
    task_name: str = "tooluse",
) -> vf.Environment:
    """Load the tool-use self-distillation environment.

    Args:
        dataset_path: Path to training data JSON (list of {example_id, messages}).
        eval_dataset_path: Path to eval data JSON (list of {example_id, prompt, golden_answer}).
        task_name: Task name for the environment (must match orchestrator config name).
    """
    train_data = _load_json(dataset_path)
    train_rows = []
    for item in train_data:
        golden_answer = _extract_golden_answer(item["messages"])
        train_rows.append(
            {
                "example_id": item["example_id"],
                "question": item["messages"][0]["content"],
                "messages": item["messages"],
                "info": {"golden_answer": golden_answer},
                "task": task_name,
            }
        )
    train_dataset = Dataset.from_list(train_rows)

    eval_dataset = None
    if eval_dataset_path:
        eval_data = _load_json(eval_dataset_path)
        eval_rows = []
        for item in eval_data:
            eval_rows.append(
                {
                    "example_id": item["example_id"],
                    "question": item["prompt"],
                    "answer": json.dumps(item["golden_answer"]),
                    "info": {"golden_answer": item["golden_answer"]},
                    "task": task_name,
                }
            )
        eval_dataset = Dataset.from_list(eval_rows)

    rubric = vf.Rubric(funcs=[_tool_call_accuracy])

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
    )


def _extract_golden_answer(messages: list[dict]) -> list[dict]:
    """Extract golden tool calls from the final assistant message in a conversation."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return _parse_tool_calls(msg.get("content", ""))
    return []


def _load_json(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def _tool_call_accuracy(completion, **kwargs) -> float:
    """Regex-match predicted tool calls against golden answers."""
    info = kwargs.get("info")
    if not info:
        return 0.0
    golden = info.get("golden_answer") if isinstance(info, dict) else None
    if not golden:
        return 0.0

    text = completion[-1].get("content", "") if completion else ""
    predicted = _parse_tool_calls(text)

    if not predicted or len(predicted) != len(golden):
        return 0.0

    correct = 0
    for pred, gold in zip(predicted, golden):
        if pred["Action"] != gold.get("Action"):
            continue
        if _compare_action_inputs(pred["Action_Input"], gold.get("Action_Input", "")):
            correct += 1

    return correct / len(golden)


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract Action/Action_Input pairs from model output."""
    text = _SPECIAL_TOKEN_RE.sub("", text)
    results = []
    for match in re.finditer(
        r"Action:\s*(\S+)\s*\n\s*Action\s*Input:\s*(.+?)(?=\nAction:|\n*$)",
        text,
        re.DOTALL,
    ):
        results.append(
            {
                "Action": match.group(1).strip(),
                "Action_Input": match.group(2).strip(),
            }
        )
    return results


def _compare_action_inputs(predicted: str, golden: str) -> bool:
    """Compare action inputs, handling JSON argument order variations."""
    try:
        return json.loads(predicted) == json.loads(golden)
    except (json.JSONDecodeError, TypeError):
        return predicted.strip() == golden.strip()
