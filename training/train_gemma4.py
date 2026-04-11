# /// script
# dependencies = ["unsloth", "trl>=0.12.0", "datasets", "trackio", "peft"]
# ///

"""Fine-tune Gemma 4 1B for emoji-only output with thinking traces."""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import json
import os

# ---------------------------------------------------------------------------
# 1. Load Gemma 4 1B with 4-bit quantisation
# ---------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gemma-4-1b-it",
    load_in_4bit=True,
    max_seq_length=1024,
)

# ---------------------------------------------------------------------------
# 2. Apply LoRA adapters
# ---------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ---------------------------------------------------------------------------
# 3. Load dataset, format as text, and create 90/10 train/eval split
# ---------------------------------------------------------------------------
dataset = load_dataset("shreyask/pantheon-ui-conversations", split="train")

def format_to_text(example):
    """Convert JSON messages to chat template text."""
    msgs = json.loads(example["messages"])
    system = example.get("system", "")
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_to_text)
dataset = dataset.remove_columns(["system", "messages"])

split = dataset.train_test_split(test_size=0.1, seed=3407)
train_dataset = split["train"]
eval_dataset = split["test"]

# ---------------------------------------------------------------------------
# 4. Configure and run SFT training
# ---------------------------------------------------------------------------
os.environ.setdefault("TRACKIO_PROJECT", "pantheon-ui")

training_args = SFTConfig(
    output_dir="output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=6,
    learning_rate=2e-4,
    warmup_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    push_to_hub=True,
    hub_model_id="shreyask/pantheon-ui-gemma4-emoji",
    hub_strategy="every_save",
    report_to="trackio",
    run_name="gemma4-emoji-sft-v1-6ep",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

trainer.train()
trainer.push_to_hub()

# --- Merge LoRA and push full model ---
print("=== Merging LoRA adapters into full model ===")
model.save_pretrained_merged(
    "merged-output",
    tokenizer,
    save_method="merged_16bit",
)

from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.upload_folder(
    repo_id="shreyask/pantheon-ui-gemma4-emoji-merged",
    folder_path="merged-output",
    commit_message=f"Gemma 4 merged model (6 epochs, loss {trainer.state.best_metric or 'N/A'})",
)
print("Merged model pushed to shreyask/pantheon-ui-gemma4-emoji-merged")
