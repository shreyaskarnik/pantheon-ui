"""
Build the decoder training dataset by inverting pantheon-ui-conversations.

Encoder direction: natural language -> <think> interpretation </think> emoji
Decoder direction: emoji -> <think> interpretation </think> natural language

The encoder dataset's thinking trace is the natural-language representation of
the message that was compressed into emoji. So for each assistant turn we emit
a new training pair where the emoji becomes the user input and the thinking
trace becomes the assistant's reconstructed natural-language output.

Together the two models form a lossy autoencoder with emoji as the bottleneck
(see Anthropic, "Natural Language Autoencoders").

Run: uv run python invert_dataset.py
"""

import json
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

SOURCE_REPO = "shreyask/pantheon-ui-conversations"
TARGET_REPO = "shreyask/pantheon-ui-decoder-conversations"

DECODER_SYSTEM_PROMPTS = [
    (
        "You are receiving compressed messages from a human consciousness that "
        "has been uploaded into a digital substrate. Its only output channel is "
        "emoji. Read the emoji sequence and reconstruct, in natural language, "
        "the message it most likely encodes. Wrap your interpretive reasoning "
        "in <think></think> tags. After the closing tag, write the reconstructed "
        "message as a single natural-language sentence or short paragraph."
    ),
    (
        "You are decoding signals from an uploaded mind whose only outward "
        "channel is emoji. The sender thinks in language but can only transmit "
        "symbols. Interpret the emoji and reconstruct, in plain English, what "
        "they were trying to say. Put your interpretation reasoning inside "
        "<think></think> tags, then write the reconstructed message."
    ),
    (
        "Emoji are the lossy bottleneck of an autoencoder: a person tried to "
        "say something, compressed it into emoji, and now you have to decompress "
        "it back into natural language. Reason about each symbol's contribution "
        "in <think></think> tags, then output the most likely original message "
        "in plain English."
    ),
]

THINK_RE = re.compile(r"^<think>([\s\S]+?)</think>\s*([\s\S]+)$")


def split_assistant(content: str) -> tuple[str, str] | None:
    m = THINK_RE.match(content.strip())
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def invert_conversation(conv: dict, prompt_idx: int) -> list[dict]:
    """Turn an encoder conversation into one or more decoder pairs.

    Each (user_text, assistant_emoji+thinking) pair becomes a decoder pair:
      - decoder user input = the emoji
      - decoder assistant output = <think> brief interpretation </think> + the
        original user_text (what the upload was responding to / about).
    """
    messages = conv.get("messages", [])
    if not messages:
        return []

    inverted_pairs: list[dict] = []
    last_user_text: str | None = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            last_user_text = content.strip()
            continue
        if role != "assistant":
            continue
        split = split_assistant(content)
        if split is None:
            continue
        thinking, emoji = split
        if not last_user_text or not emoji:
            continue
        decoder_assistant = (
            f"<think>{thinking}</think>\n{last_user_text}"
        )
        inverted_pairs.append(
            {
                "system": DECODER_SYSTEM_PROMPTS[prompt_idx % len(DECODER_SYSTEM_PROMPTS)],
                "messages": [
                    {"role": "user", "content": emoji},
                    {"role": "assistant", "content": decoder_assistant},
                ],
            }
        )

    return inverted_pairs


def main() -> None:
    print(f"Loading source dataset: {SOURCE_REPO}")
    src = load_dataset(SOURCE_REPO, split="train")
    print(f"Loaded {len(src)} encoder conversations.")

    inverted: list[dict] = []
    for i, row in enumerate(src):
        msgs = row["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        conv = {"system": row.get("system", ""), "messages": msgs}
        inverted.extend(invert_conversation(conv, prompt_idx=i))

    print(f"Produced {len(inverted)} decoder pairs.")

    rows = [
        {
            "system": item["system"],
            "messages": json.dumps(item["messages"], ensure_ascii=False),
        }
        for item in inverted
    ]
    ds = Dataset.from_list(rows)
    print(f"Decoder dataset: {ds}")

    print(f"Pushing to {TARGET_REPO}…")
    ds.push_to_hub(
        TARGET_REPO,
        private=False,
        commit_message="Inverted pantheon-ui-conversations for decoder training",
    )

    card_path = Path(__file__).parent / "card" / "DECODER.md"
    if card_path.exists():
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=TARGET_REPO,
            repo_type="dataset",
            commit_message="Add decoder dataset card",
        )

    print(f"Done! https://huggingface.co/datasets/{TARGET_REPO}")


if __name__ == "__main__":
    main()
