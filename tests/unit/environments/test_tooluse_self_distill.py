import json
import tempfile
from pathlib import Path

import pytest

from tooluse_self_distill import (
    _compare_action_inputs,
    _parse_tool_calls,
    _tool_call_accuracy,
    load_environment,
)


@pytest.fixture
def train_data_path(tmp_path):
    data = [
        {
            "example_id": 0,
            "messages": [
                {"role": "user", "content": "Use tool X to do Y."},
                {"role": "assistant", "content": "Action: toolX\nAction Input: {\"arg\": 1}"},
            ],
        },
        {
            "example_id": 1,
            "messages": [
                {"role": "user", "content": "Use tool Z to do W."},
                {"role": "assistant", "content": "Action: toolZ\nAction Input: {\"key\": \"val\"}"},
            ],
        },
    ]
    path = tmp_path / "train.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def eval_data_path(tmp_path):
    data = [
        {
            "example_id": 0,
            "prompt": "Use tool X to do Y.",
            "instruction": "Do Y.",
            "golden_answer": [{"Action": "toolX", "Action_Input": '{"arg": 1}'}],
        },
    ]
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_load_train_only(train_data_path):
    env = load_environment(dataset_path=train_data_path)
    dataset = env.get_dataset()

    assert "example_id" in dataset.column_names
    assert "prompt" in dataset.column_names
    assert "task" in dataset.column_names
    assert "messages" in dataset.column_names
    assert len(dataset) == 2

    row = dataset[0]
    assert row["example_id"] == 0
    assert row["task"] == "tooluse"
    assert len(row["messages"]) == 2
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "assistant"


def test_load_with_eval(train_data_path, eval_data_path):
    env = load_environment(
        dataset_path=train_data_path,
        eval_dataset_path=eval_data_path,
    )
    eval_dataset = env.get_eval_dataset()

    assert eval_dataset is not None
    assert len(eval_dataset) == 1
    assert "example_id" in eval_dataset.column_names
    assert "prompt" in eval_dataset.column_names
    assert "answer" in eval_dataset.column_names


def test_custom_task_name(train_data_path):
    env = load_environment(dataset_path=train_data_path, task_name="my-task")
    dataset = env.get_dataset()
    assert all(row["task"] == "my-task" for row in dataset)


def test_messages_preserved_for_self_distill(train_data_path):
    """Messages must be accessible for self_distill_context.build_teacher_prompt."""
    env = load_environment(dataset_path=train_data_path)
    dataset = env.get_dataset()

    for row in dataset:
        messages = row["messages"]
        assert isinstance(messages, list)
        assert len(messages) >= 2
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles
        for m in messages:
            assert isinstance(m["content"], str)


def test_parse_tool_calls():
    text = "Thought: I need to call toolX\nAction: toolX\nAction Input: {\"arg\": 1}"
    calls = _parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["Action"] == "toolX"
    assert calls[0]["Action_Input"] == '{"arg": 1}'


def test_parse_tool_calls_multiple():
    text = (
        "Thought: first\nAction: toolA\nAction Input: {}\n"
        "Action: toolB\nAction Input: {\"x\": 2}"
    )
    calls = _parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["Action"] == "toolA"
    assert calls[1]["Action"] == "toolB"


def test_compare_action_inputs_json_order():
    assert _compare_action_inputs('{"b": 2, "a": 1}', '{"a": 1, "b": 2}')


def test_compare_action_inputs_string_fallback():
    assert _compare_action_inputs("plain text", "plain text")
    assert not _compare_action_inputs("plain text", "other text")


def test_tool_call_accuracy_no_info():
    assert _tool_call_accuracy(completion=[{"role": "assistant", "content": "hi"}]) == 0.0


def test_tool_call_accuracy_correct():
    completion = [{"role": "assistant", "content": "Action: toolX\nAction Input: {\"arg\": 1}"}]
    info = {"golden_answer": [{"Action": "toolX", "Action_Input": '{"arg": 1}'}]}
    assert _tool_call_accuracy(completion, info=info) == 1.0


def test_tool_call_accuracy_wrong_action():
    completion = [{"role": "assistant", "content": "Action: toolY\nAction Input: {\"arg\": 1}"}]
    info = {"golden_answer": [{"Action": "toolX", "Action_Input": '{"arg": 1}'}]}
    assert _tool_call_accuracy(completion, info=info) == 0.0
