"""
Build the decoder training dataset by inverting pantheon-ui-conversations.

Encoder direction: natural language -> <think> interpretation </think> emoji
Decoder direction: emoji -> <think> interpretation </think> natural language

The encoder dataset's thinking trace is the natural-language representation of
the message that was compressed into emoji. So for each assistant turn we emit
a new training pair where the emoji becomes the user input and the thinking
trace becomes the assistant's reconstructed natural-language output.

Together the two models form a round-trip translator: text -> emoji -> text,
with emoji as a discrete, human-legible intermediate. Inspired by Anthropic's
"Natural Language Autoencoders".

Run: uv run python invert_dataset.py
     uv run python invert_dataset.py --dry-run   # preview without pushing
"""

import argparse
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
        "Emoji are a lossy intermediate: a person tried to say something, "
        "compressed it into emoji, and now you have to decompress it back into "
        "natural language. Reason about each symbol's contribution in "
        "<think></think> tags, then output the most likely original message "
        "in plain English."
    ),
]

# Generic, perspective-correct think placeholders. We deliberately don't use
# the encoder's think trace as the decoder target: the encoder thought from
# the upload's perspective ("I feel her rage…"), but the decoder's job is to
# decode emoji into the most likely original message, which is a different
# speaker's words. Using the encoder's CoT scrambles the perspective.
#
# These placeholders are short, vary by system prompt, and frame the reasoning
# from the decoder's actual perspective ("reading these symbols…"). The model
# will learn to emit a templated think block + a unique reconstruction.
DECODER_THINK_PLACEHOLDERS = [
    "Reading these emoji as compressed signal — each symbol carries a fragment of mood, action, or referent. Reconstructing the most plausible plain-language message the upload was trying to send.",
    "Decoding one emoji at a time, weighing what the sender most likely intended, and assembling the natural-language equivalent of the original utterance.",
    "These emoji are a lossy compression of an utterance. Inferring the most likely original words from the symbols and the implicit emotional context.",
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
      - decoder user input = the emoji (encoder's output)
      - decoder assistant output = <think> generic placeholder </think> + the
        original user_text (what the upload was responding to / about).

    The encoder's `<think>` content is intentionally discarded — see the
    comment on DECODER_THINK_PLACEHOLDERS above.
    """
    messages = conv.get("messages", [])
    if not messages:
        return []

    system_prompt = DECODER_SYSTEM_PROMPTS[prompt_idx % len(DECODER_SYSTEM_PROMPTS)]
    think_placeholder = DECODER_THINK_PLACEHOLDERS[prompt_idx % len(DECODER_THINK_PLACEHOLDERS)]

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
        _, emoji = split
        if not last_user_text or not emoji:
            continue
        decoder_assistant = (
            f"<think>{think_placeholder}</think>\n{last_user_text}"
        )
        inverted_pairs.append(
            {
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": emoji},
                    {"role": "assistant", "content": decoder_assistant},
                ],
            }
        )

    return inverted_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print first 3 inverted pairs and counts; do not push to HF.",
    )
    args = parser.parse_args()

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

    if args.dry_run:
        print("\n=== DRY RUN — first 3 decoder pairs ===")
        for i, pair in enumerate(inverted[:3]):
            print(f"\n--- pair {i + 1} ---")
            print(f"system: {pair['system'][:120]}…")
            for m in pair["messages"]:
                content = m["content"]
                preview = content if len(content) < 400 else content[:400] + "…"
                print(f"[{m['role']}] {preview}")
        prompt_counts: dict[str, int] = {}
        for p in inverted:
            prompt_counts[p["system"][:40]] = prompt_counts.get(p["system"][:40], 0) + 1
        print("\nPrompt distribution (first 40 chars → count):")
        for k, v in prompt_counts.items():
            print(f"  {v:>4}  {k}…")
        print(f"\nTotal pairs: {len(inverted)}. Skipping push (dry run).")
        return

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
