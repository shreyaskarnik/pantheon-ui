"""
Generate multi-turn training conversations using the Claude API.
Uses asyncio for parallel API calls across categories.
Reads seed examples from seed_examples.json and writes to dataset.jsonl.
"""

import asyncio
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
MAX_CONCURRENT = 5  # parallel API calls

# ── Validation ────────────────────────────────────────────────────────────────
THINK_RE = re.compile(r"^<think>([\s\S]+?)</think>\s*([\s\S]+)$")


def is_emoji_only(text: str) -> bool:
    """Return True if text contains no latin characters (ignoring ellipsis)."""
    cleaned = text.strip()
    if not cleaned:
        return False
    no_ellipsis = cleaned.replace("...", "").replace("\u2026", "")
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
        if msg.get("role") == "assistant":
            ok, reason = validate_assistant_message(msg.get("content", ""))
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

CRITICAL: The thinking trace MUST be concise (1-3 sentences, 15-40 words). \
Do NOT write long paragraphs of internal monologue. Keep it short, emotional, \
and human — like someone quickly thinking before responding.

## Category: {category}

Category description: {description}

## Seed examples for this category:

{seed_examples}

## Your task

Generate exactly {n} new conversations for the **{category}** category. \
Each conversation should:
1. Have 2–6 turns (user/assistant pairs).
2. Have the assistant always respond with <think>…</think> followed by emoji only.
3. Thinking traces should be SHORT (1-3 sentences, 15-40 words). Not essays.
4. Emoji responses should be evocative and match the thinking.
5. Vary the emotional tone, length, and situation across conversations.
6. Stay true to the "{category}" theme.

Return ONLY a JSON array of conversation objects. Each object has a "messages" key \
containing an array of {{"role": "user"|"assistant", "content": "…"}} objects. \
No extra keys, no markdown fences, no explanation — just the raw JSON array.
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
    seed_examples_str = json.dumps(seeds[:4], indent=2, ensure_ascii=False)
    return GENERATION_PROMPT_TEMPLATE.format(
        n=n,
        category=category,
        description=CATEGORY_DESCRIPTIONS.get(category, category),
        seed_examples=seed_examples_str,
    )


# ── Generation ────────────────────────────────────────────────────────────────

def parse_conversations(raw: str) -> list[dict]:
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
    m = re.search(r"\[[\s\S]+\]", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []


async def generate_batch(
    client: anthropic.AsyncAnthropic,
    category: str,
    seeds: list[dict],
    system_prompts: list[str],
    n: int = BATCH_SIZE,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[list[dict], int, int]:
    """Generate a batch of conversations. Returns (valid_convs, input_tokens, output_tokens)."""
    prompt = build_prompt(category, seeds, n)

    async with semaphore if semaphore else asyncio.Lock():
        response = await client.messages.create(
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
        system_prompt = random.choice(system_prompts)
        full_conv = {"system": system_prompt, "messages": conv.get("messages", [])}
        ok, reason = validate_conversation(full_conv)
        if ok:
            valid.append(full_conv)

    return valid, input_tokens, output_tokens


async def generate_category(
    client: anthropic.AsyncAnthropic,
    category: str,
    seed_convs: list[dict],
    system_prompts: list[str],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Generate all conversations for a single category."""
    all_valid = []
    batches_done = 0

    while len(all_valid) < TARGET_PER_CATEGORY and batches_done < MAX_BATCHES_PER_CATEGORY:
        remaining = TARGET_PER_CATEGORY - len(all_valid)
        batch_n = min(BATCH_SIZE, remaining)

        for attempt in range(1, 3):
            try:
                valid_convs, inp_tok, out_tok = await generate_batch(
                    client, category, seed_convs, system_prompts, n=batch_n, semaphore=semaphore
                )
                print(
                    f"  [{category}] batch {batches_done+1} attempt {attempt}: "
                    f"{len(valid_convs)}/{batch_n} valid | "
                    f"{inp_tok}in/{out_tok}out"
                )
                all_valid.extend(valid_convs)
                break
            except Exception as e:
                print(f"  [{category}] batch {batches_done+1} attempt {attempt} ERROR: {e}")
                if attempt >= 2:
                    print(f"  [{category}] skipping batch after 2 failures")

        batches_done += 1

    print(f"  [{category}] DONE: {len(all_valid)} conversations")
    return all_valid


async def main():
    with open(SEED_FILE) as f:
        seeds_data = json.load(f)

    system_prompts: list[str] = seeds_data["system_prompts"]
    categories: dict[str, list[dict]] = seeds_data["categories"]

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    print(f"Generating {TARGET_PER_CATEGORY} conversations per category")
    print(f"Categories: {len(categories)} | Max concurrent: {MAX_CONCURRENT}")
    print(f"{'='*60}")

    # Launch all categories concurrently (throttled by semaphore)
    tasks = {
        category: asyncio.create_task(
            generate_category(client, category, seed_convs, system_prompts, semaphore)
        )
        for category, seed_convs in categories.items()
    }

    # Gather results
    results = {}
    for category, task in tasks.items():
        results[category] = await task

    # Write all results
    total = 0
    with open(OUTPUT_FILE, "w") as f:
        for category, convs in results.items():
            for conv in convs:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            total += len(convs)

    print(f"\n{'='*60}")
    print(f"FINISHED: {total} conversations written to {OUTPUT_FILE}")
    for category, convs in results.items():
        print(f"  {category}: {len(convs)}")


if __name__ == "__main__":
    asyncio.run(main())
