"""Generic HuggingFace messages environment for OPSD training.

Loads any HF dataset that has a ``messages`` column (list of
``{role, content}`` dicts) and exposes it as a ``vf.SingleTurnEnv``.

Supports two common message formats:
  - [system, user, assistant]  – system prompt extracted automatically
  - [user, assistant]          – no system prompt

Usage in a TOML config::

    [[orchestrator.env]]
    id = "hf-messages"
    name = "hf_messages"
    args = { dataset_name = "org/dataset" }
"""

import verifiers as vf
from datasets import load_dataset


def load_environment(
    dataset_name: str,
    dataset_split: str = "train",
    messages_column: str = "messages",
    **kwargs,
) -> vf.Environment:
    raw_dataset = load_dataset(dataset_name, split=dataset_split)

    first_messages = raw_dataset[0][messages_column]
    system_prompt = None
    if first_messages[0]["role"] == "system":
        system_prompt = first_messages[0]["content"]
        user_idx, assistant_idx = 1, 2
    else:
        user_idx, assistant_idx = 0, 1

    train_dataset = raw_dataset.map(
        lambda x: {
            "question": x[messages_column][user_idx]["content"],
            "answer": x[messages_column][assistant_idx]["content"],
        }
    )

    rubric = vf.Rubric()
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
    )
