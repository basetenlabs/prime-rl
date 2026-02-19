import pytest

from prime_rl.orchestrator.self_distill_context import build_example_lookup, build_teacher_prompt


def test_build_teacher_prompt_renders_transcript_and_demonstration():
    dataset = [
        {
            "example_id": 7,
            "messages": [
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
        }
    ]
    lookup = build_example_lookup(dataset)
    prompt = build_teacher_prompt(lookup, 7)

    assert "<USER>\nQ1\n</USER>" in prompt
    assert "<ASSISTANT>\nA1\n</ASSISTANT>" in prompt
    assert "<USER>\nQ2\n</USER>" in prompt
    assert "<Demonstration>\nA2\n" in prompt


def test_build_teacher_prompt_with_system_message():
    dataset = [
        {
            "example_id": 11,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
        }
    ]
    lookup = build_example_lookup(dataset)
    prompt = build_teacher_prompt(lookup, 11)

    assert "<SYSTEM>\nYou are a helpful assistant.\n</SYSTEM>" in prompt
    assert "<USER>\nWhat is 2+2?\n</USER>" in prompt
    assert "<Demonstration>\n4\n" in prompt


def test_build_teacher_prompt_rejects_non_text_content():
    dataset = [
        {
            "example_id": 9,
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x"}}]},
                {"role": "assistant", "content": "A"},
            ],
        }
    ]
    lookup = build_example_lookup(dataset)
    with pytest.raises(ValueError, match="example_id=9"):
        build_teacher_prompt(lookup, 9)
