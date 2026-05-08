# /// script
# dependencies = ["unsloth", "trl>=0.12.0", "datasets", "trackio", "peft"]
# ///

"""
Fine-tune the decoder half of the pantheon-ui autoencoder.

Encoder: natural language -> emoji (shreyask/pantheon-ui-lfm25-emoji)
Decoder: emoji -> natural language (this script)

Together: a lossy autoencoder with emoji as the bottleneck.
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "LiquidAI/LFM2.5-1.2B-Thinking",
    load_in_4bit=True,
    max_seq_length=1024,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "out_proj", "in_proj",
        "w1", "w2", "w3",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

import json

dataset = load_dataset("shreyask/pantheon-ui-decoder-conversations", split="train")


def format_to_text(example):
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

training_args = SFTConfig(
    output_dir="output-decoder",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=6,
    learning_rate=2e-4,
    warmup_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    push_to_hub=True,
    hub_model_id="shreyask/pantheon-ui-decoder-lfm25",
    hub_strategy="every_save",
    report_to="trackio",
    run_name="lfm25-decoder-sft-v1-6ep",
)

import os
os.environ.setdefault("TRACKIO_PROJECT", "pantheon-ui-decoder")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

trainer.train()
trainer.push_to_hub()

print("=== Merging LoRA adapters into full decoder model ===")
model.save_pretrained_merged(
    "merged-decoder-output",
    tokenizer,
    save_method="merged_16bit",
)

from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.upload_folder(
    repo_id="shreyask/pantheon-ui-decoder-lfm25-merged",
    folder_path="merged-decoder-output",
    commit_message=f"Merged decoder model (6 epochs, loss {trainer.state.best_metric or 'N/A'})",
)
print("Merged decoder pushed to shreyask/pantheon-ui-decoder-lfm25-merged")
