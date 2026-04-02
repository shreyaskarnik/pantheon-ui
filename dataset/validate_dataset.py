"""
Validate dataset.jsonl and print statistics.
"""

import json
import re
import statistics
from pathlib import Path
from collections import defaultdict

DATASET_FILE = Path(__file__).parent / "dataset.jsonl"

THINK_RE = re.compile(r"^<think>([\s\S]+?)</think>\s*([\s\S]+)$")


def is_emoji_only(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    no_ellipsis = cleaned.replace("...", "").replace("…", "")
    if re.search(r"[a-zA-Z]", no_ellipsis):
        return False
    return True


def validate_assistant_message(content: str) -> tuple[bool, str]:
    m = THINK_RE.match(content.strip())
    if not m:
        return False, "missing <think>...</think> block or emoji portion"
    think_text, emoji_text = m.group(1), m.group(2)
    if len(think_text.strip()) < 10:
        return False, f"thinking trace too short ({len(think_text.strip())} chars)"
    if not is_emoji_only(emoji_text):
        return False, "emoji portion contains latin characters"
    return True, ""


def validate_conversation(conv: dict) -> tuple[bool, str]:
    if "system" not in conv or not conv["system"]:
        return False, "missing system prompt"
    messages = conv.get("messages", [])
    if not messages:
        return False, "no messages"
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            ok, reason = validate_assistant_message(content)
            if not ok:
                return False, f"message[{i}]: {reason}"
    return True, ""


def count_turns(conv: dict) -> int:
    """Number of user/assistant pairs."""
    messages = conv.get("messages", [])
    return sum(1 for m in messages if m.get("role") == "user")


def thinking_word_count(content: str) -> int:
    m = THINK_RE.match(content.strip())
    if not m:
        return 0
    return len(m.group(1).split())


def emoji_char_count(content: str) -> int:
    m = THINK_RE.match(content.strip())
    if not m:
        return 0
    return len(m.group(2).strip())


def main():
    if not DATASET_FILE.exists():
        print(f"ERROR: {DATASET_FILE} does not exist.")
        return

    conversations = []
    with open(DATASET_FILE) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
                conversations.append((i, conv))
            except json.JSONDecodeError as e:
                print(f"Line {i}: JSON parse error: {e}")

    total = len(conversations)
    print(f"\n{'='*60}")
    print(f"DATASET VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Total lines: {total}")

    valid_convs = []
    invalid_convs = []
    invalid_reasons = defaultdict(int)

    for line_no, conv in conversations:
        ok, reason = validate_conversation(conv)
        if ok:
            valid_convs.append(conv)
        else:
            invalid_convs.append((line_no, reason))
            invalid_reasons[reason] += 1

    print(f"Valid:   {len(valid_convs)} ({100*len(valid_convs)/total:.1f}%)" if total else "Valid: 0")
    print(f"Invalid: {len(invalid_convs)} ({100*len(invalid_convs)/total:.1f}%)" if total else "Invalid: 0")

    if invalid_reasons:
        print(f"\nInvalid reasons breakdown:")
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1]):
            print(f"  [{count:4d}] {reason}")

    if valid_convs:
        # Thinking trace word counts
        think_words = []
        emoji_lengths = []
        turn_counts = []

        for conv in valid_convs:
            turn_counts.append(count_turns(conv))
            for msg in conv.get("messages", []):
                if msg.get("role") == "assistant":
                    think_words.append(thinking_word_count(msg["content"]))
                    emoji_lengths.append(emoji_char_count(msg["content"]))

        def stats(data: list[float], label: str):
            if not data:
                return
            print(f"\n{label}:")
            print(f"  count:  {len(data)}")
            print(f"  min:    {min(data):.0f}")
            print(f"  max:    {max(data):.0f}")
            print(f"  mean:   {statistics.mean(data):.1f}")
            print(f"  median: {statistics.median(data):.1f}")
            if len(data) > 1:
                print(f"  stdev:  {statistics.stdev(data):.1f}")

        stats(think_words, "Thinking trace length (words)")
        stats(emoji_lengths, "Emoji response length (chars)")
        stats(turn_counts, "Turns per conversation (user messages)")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
