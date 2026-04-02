# /// script
# dependencies = ["unsloth", "trl>=0.12.0", "datasets", "trackio", "peft"]
# ///

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# 1. Load base model with 4-bit quantisation
# ---------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    "LiquidAI/LFM2.5-1.2B-Thinking",
    load_in_4bit=True,
    max_seq_length=1024,
)

# ---------------------------------------------------------------------------
# 2. Apply LoRA adapters (LFM2-specific target modules)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. Load dataset and create 90/10 train/eval split
# ---------------------------------------------------------------------------
dataset = load_dataset("shreyask/pantheon-ui-conversations", split="train")
split = dataset.train_test_split(test_size=0.1, seed=3407)
train_dataset = split["train"]
eval_dataset = split["test"]

# ---------------------------------------------------------------------------
# 4. Configure and run SFT training
# ---------------------------------------------------------------------------
training_args = SFTConfig(
    output_dir="output",
    # Batch / accumulation
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    # Schedule
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=50,
    # Evaluation
    eval_strategy="steps",
    eval_steps=50,
    # Hub
    push_to_hub=True,
    hub_model_id="shreyask/pantheon-ui-lfm25-emoji",
    hub_strategy="every_save",
    # Logging / tracking
    report_to="trackio",
    run_name="lfm25-emoji-sft-v1",
    # Trackio project label (passed via environment or SFTConfig extra kwargs)
    # trackio reads HF_PROJECT / TRACKIO_PROJECT from env; set it explicitly too
)

# Trackio project name (read by trackio from this env-style attribute when present)
import os
os.environ.setdefault("TRACKIO_PROJECT", "pantheon-ui")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

trainer.train()
trainer.push_to_hub()
