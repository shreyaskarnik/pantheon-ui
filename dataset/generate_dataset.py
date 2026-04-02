"""
Generate multi-turn training conversations using the Claude API.
Reads seed examples from seed_examples.json and writes to dataset.jsonl.
"""

import json
import re
import random
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
SEED_FILE = Path(__file__).parent / "seed_examples.json"
OUTPUT_FILE = Path(__file__).parent / "dataset.jsonl"
MODEL = "claude-sonnet-4-20250514"
TARGET_PER_CATEGORY = 50
BATCH_SIZE = 10
MAX_BATCHES_PER_CATEGORY = 8

# ── Validation ────────────────────────────────────────────────────────────────
THINK_RE = re.compile(r"^<think>([\s\S]+?)</think>\s*([\s\S]+)$")


def is_emoji_only(text: str) -> bool:
    """Return True if text contains no latin characters (ignoring ellipsis)."""
    cleaned = text.strip()
    if not cleaned:
        return False
    no_ellipsis = cleaned.replace("...", "").replace("…", "")
    if re.search(r"[a-zA-Z]", no_ellipsis):
        return False
    return True


def validate_assistant_message(content: str) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    Assistant messages must be: <think>…</think>\n<emoji-only>
    """
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
    """Validate a full conversation dict (with 'system' and 'messages' keys)."""
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
                return False, f"message[{i}] invalid: {reason}"
    return True, ""


# ── Prompt building ───────────────────────────────────────────────────────────

GENERATION_PROMPT_TEMPLATE = """\
You are helping generate training data. I will show you example multi-turn \
conversations from a specific category, and I want you to generate {n} NEW, \
varied conversations in the same style and format.

## Context

The AI in these conversations is an uploaded human consciousness that can only \
communicate via emoji. Its internal monologue is wrapped in <think></think> tags, \
and its actual response is emoji only — never letters, never punctuation outside \
the emoji, never numbers as text.

## Category: {category}

Category description: {description}

## Seed examples for this category:

{seed_examples}

## Your task

Generate exactly {n} new conversations for the **{category}** category. \
Each conversation should:
1. Have 2–6 turns (user/assistant pairs).
2. Have the assistant always respond with <think>…</think> followed by emoji only.
3. Thinking traces should be rich and expressive (20+ words).
4. Emoji responses should be evocative and match the thinking.
5. Vary the emotional tone, length, and situation across conversations.
6. Stay true to the "{category}" theme.

Return ONLY a JSON array of conversation objects. Each object has a "messages" key \
containing an array of {{"role": "user"|"assistant", "content": "…"}} objects. \
No extra keys, no markdown fences, no explanation — just the raw JSON array.

Example output structure:
[
  {{
    "messages": [
      {{"role": "user", "content": "…"}},
      {{"role": "assistant", "content": "<think>…</think>\\n<emoji>"}}
    ]
  }},
  …
]
"""

CATEGORY_DESCRIPTIONS = {
    "emotional": "Deep emotional exchanges — grief, love, longing, joy, and anger",
    "yes_no": "Simple yes/no questions answered expressively through emoji",
    "warnings": "The consciousness tries to warn or alert the user about something",
    "spatial": "Describing location, movement, or spatial concepts through emoji",
    "temporal": "Time-related exchanges — past, future, duration, waiting",
    "informational": "Answering factual or practical questions purely through emoji",
    "storytelling": "Narrating a memory or telling a story using emoji sequences",
    "meta_existential": "Reflecting on the nature of being uploaded, existing digitally",
    "frustration": "Expressing frustration with the emoji limitation or the situation",
    "playful": "Lighthearted, playful, or funny exchanges",
    "complex_abstract": "Abstract or philosophical ideas expressed through symbol-chains",
    "reassurance": "The consciousness comforting or reassuring the user",
}


def build_prompt(category: str, seeds: list[dict], n: int) -> str:
    seed_examples_str = json.dumps(seeds[:4], indent=2)  # show up to 4 seeds
    return GENERATION_PROMPT_TEMPLATE.format(
        n=n,
        category=category,
        description=CATEGORY_DESCRIPTIONS.get(category, category),
        seed_examples=seed_examples_str,
    )


# ── Generation ────────────────────────────────────────────────────────────────

def parse_conversations(raw: str) -> list[dict]:
    """Try to extract a JSON array from the raw response text."""
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # Try to find the array inside the response
    m = re.search(r"\[[\s\S]+\]", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []


def generate_batch(
    client: anthropic.Anthropic,
    category: str,
    seeds: list[dict],
    system_prompts: list[str],
    n: int = BATCH_SIZE,
) -> tuple[list[dict], int, int]:
    """
    Generate a batch of conversations. Returns (valid_convs, input_tokens, output_tokens).
    """
    prompt = build_prompt(category, seeds, n)
    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    raw_text = response.content[0].text
    parsed = parse_conversations(raw_text)

    valid = []
    for conv in parsed:
        if not isinstance(conv, dict):
            continue
        # Attach a random system prompt
        system_prompt = random.choice(system_prompts)
        full_conv = {"system": system_prompt, "messages": conv.get("messages", [])}
        ok, reason = validate_conversation(full_conv)
        if ok:
            valid.append(full_conv)
        else:
            print(f"    [skip] invalid conversation: {reason}")

    return valid, input_tokens, output_tokens


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(SEED_FILE) as f:
        seeds_data = json.load(f)

    system_prompts: list[str] = seeds_data["system_prompts"]
    categories: dict[str, list[dict]] = seeds_data["categories"]

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    total_conversations = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # Open output in append mode so reruns add to existing data
    out_file = open(OUTPUT_FILE, "a")

    try:
        for category, seed_convs in categories.items():
            print(f"\n{'='*60}")
            print(f"Category: {category}  (seeds: {len(seed_convs)})")
            print(f"{'='*60}")

            category_count = 0
            batches_done = 0

            while category_count < TARGET_PER_CATEGORY and batches_done < MAX_BATCHES_PER_CATEGORY:
                remaining = TARGET_PER_CATEGORY - category_count
                batch_n = min(BATCH_SIZE, remaining)
                attempt = 0
                success = False

                while attempt < 2:
                    attempt += 1
                    print(f"  Batch {batches_done + 1} (attempt {attempt}): requesting {batch_n} conversations...")
                    try:
                        valid_convs, inp_tok, out_tok = generate_batch(
                            client, category, seed_convs, system_prompts, n=batch_n
                        )
                        total_input_tokens += inp_tok
                        total_output_tokens += out_tok
                        print(
                            f"    tokens: {inp_tok} in / {out_tok} out  |  "
                            f"valid: {len(valid_convs)}/{batch_n}"
                        )
                        for conv in valid_convs:
                            out_file.write(json.dumps(conv, ensure_ascii=False) + "\n")
                        out_file.flush()
                        category_count += len(valid_convs)
                        success = True
                        break
                    except Exception as e:
                        print(f"    ERROR on attempt {attempt}: {e}")
                        if attempt >= 2:
                            print("    Skipping batch after 2 failed attempts.")

                if not success:
                    batches_done += 1
                    continue

                batches_done += 1
                total_conversations += len(valid_convs) if success else 0

                # Running totals
                est_cost = (total_input_tokens / 1_000_000 * 3.0) + (
                    total_output_tokens / 1_000_000 * 15.0
                )
                print(
                    f"  [{category}] {category_count}/{TARGET_PER_CATEGORY} so far  |  "
                    f"total convs: {total_conversations}  |  "
                    f"tokens: {total_input_tokens}in/{total_output_tokens}out  |  "
                    f"est. cost: ${est_cost:.3f}"
                )

            print(f"  Done with {category}: {category_count} conversations generated.")

    finally:
        out_file.close()

    print(f"\n{'='*60}")
    print(f"FINISHED")
    print(f"  Total conversations written: {total_conversations}")
    print(f"  Total input tokens:  {total_input_tokens:,}")
    print(f"  Total output tokens: {total_output_tokens:,}")
    est_cost = (total_input_tokens / 1_000_000 * 3.0) + (
        total_output_tokens / 1_000_000 * 15.0
    )
    print(f"  Estimated cost: ${est_cost:.4f}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
