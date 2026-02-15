from numbers import Integral
from typing import Any


_ROLE_TO_TAG = {
    "user": "USER",
    "assistant": "ASSISTANT",
    "tool": "TOOL",
}


def build_example_lookup(train_dataset: Any) -> dict[int, dict[str, Any]]:
    example_lookup: dict[int, dict[str, Any]] = {}
    for example in train_dataset:
        if not isinstance(example, dict):
            raise ValueError(f"Expected dataset rows to be dict-like, got {type(example)}")
        example_id = example.get("example_id")
        if not isinstance(example_id, Integral):
            raise ValueError(f"Expected integer example_id, got {example_id!r}")
        example_id = int(example_id)
        if example_id in example_lookup:
            raise ValueError(f"Duplicate example_id={example_id} in dataset")
        example_lookup[example_id] = dict(example)
    return example_lookup


def build_teacher_prompt(example_lookup: dict[int, dict[str, Any]], example_id: int) -> str:
    if example_id not in example_lookup:
        raise ValueError(f"Failed to build self-distill prompt: example_id={example_id} not found in dataset")
    example = example_lookup[example_id]
    messages = example.get("messages")
    transcript, demonstration = _extract_transcript_and_demonstration(messages, example_id)
    return (
        "<Question>\n"
        f"{transcript}\n"
        "This is an example for a response to the question:\n"
        "<Demonstration>\n"
        f"{demonstration}\n"
        "Now answer with a response of your own, including the thinking process:"
    )


def _extract_transcript_and_demonstration(messages: Any, example_id: int) -> tuple[str, str]:
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError(f"example_id={example_id}: missing or invalid `messages` field")

    final_assistant_idx = None
    for idx, message in enumerate(messages):
        role = _extract_role(message, example_id, idx)
        if role == "assistant":
            final_assistant_idx = idx

    if final_assistant_idx is None:
        raise ValueError(f"example_id={example_id}: no assistant message found for demonstration")

    final_user_idx = None
    for idx in range(final_assistant_idx - 1, -1, -1):
        role = _extract_role(messages[idx], example_id, idx)
        if role == "user":
            final_user_idx = idx
            break

    if final_user_idx is None:
        raise ValueError(
            f"example_id={example_id}: no user message found before final assistant message at index {final_assistant_idx}"
        )

    transcript = _render_transcript(messages[: final_user_idx + 1], example_id)
    demonstration = _extract_text_content(messages[final_assistant_idx], example_id, final_assistant_idx)
    return transcript, demonstration


def _extract_role(message: Any, example_id: int, role_idx: int) -> str:
    if not isinstance(message, dict):
        raise ValueError(f"example_id={example_id}, role_idx={role_idx}: message must be a dict")
    role = message.get("role")
    if role not in _ROLE_TO_TAG:
        raise ValueError(
            f"example_id={example_id}, role_idx={role_idx}: unsupported role {role!r}; expected one of {sorted(_ROLE_TO_TAG)}"
        )
    return role


def _extract_text_content(message: dict[str, Any], example_id: int, role_idx: int) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                raise ValueError(
                    f"example_id={example_id}, role_idx={role_idx}, content_idx={part_idx}: expected dict content part"
                )
            if part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise ValueError(
                    f"example_id={example_id}, role_idx={role_idx}, content_idx={part_idx}: non-text content is not supported in self-distill mode"
                )
            text_parts.append(part["text"])
        return "".join(text_parts)
    raise ValueError(
        f"example_id={example_id}, role_idx={role_idx}: non-text content is not supported in self-distill mode"
    )


def _render_transcript(messages: list[dict[str, Any]], example_id: int) -> str:
    rendered_messages: list[str] = []
    for idx, message in enumerate(messages):
        role = _extract_role(message, example_id, idx)
        text = _extract_text_content(message, example_id, idx)
        tag = _ROLE_TO_TAG[role]
        rendered_messages.append(f"<{tag}>\n{text}\n</{tag}>")
    return "\n".join(rendered_messages)
