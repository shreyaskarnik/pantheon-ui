"""
Push dataset.jsonl to the Hugging Face Hub as shreyask/pantheon-ui-conversations.
Requires HF_TOKEN in the environment (or dataset/.env).
"""

import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from datasets import Dataset
from huggingface_hub import HfApi

DATASET_FILE = Path(__file__).parent / "dataset.jsonl"
HF_REPO_ID = "shreyask/pantheon-ui-conversations"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_for_hf(conversations: list[dict]) -> list[dict]:
    """
    Convert each conversation into a flat HF-friendly row:
      - system: str
      - messages: list[dict]  (kept as-is for easy SFT use)
    """
    rows = []
    for conv in conversations:
        rows.append(
            {
                "system": conv.get("system", ""),
                "messages": json.dumps(conv.get("messages", []), ensure_ascii=False),
            }
        )
    return rows


def main():
    if not DATASET_FILE.exists():
        print(f"ERROR: {DATASET_FILE} does not exist. Run generate_dataset.py first.")
        return

    print(f"Loading {DATASET_FILE}…")
    conversations = load_jsonl(DATASET_FILE)
    print(f"Loaded {len(conversations)} conversations.")

    rows = flatten_for_hf(conversations)
    ds = Dataset.from_list(rows)
    print(f"Dataset: {ds}")

    print(f"\nPushing to {HF_REPO_ID}…")
    ds.push_to_hub(
        HF_REPO_ID,
        private=False,
        commit_message="Update pantheon-ui-conversations dataset",
    )
    print(f"Done! https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
