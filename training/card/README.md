---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
  - emoji
  - thinking
  - chain-of-thought
  - pantheon
  - unsloth
  - lfm
  - webgpu
  - transformers.js
base_model: LiquidAI/LFM2.5-1.2B-Thinking
datasets:
  - shreyask/pantheon-ui-conversations
---

# Pantheon UI — LFM2.5 Emoji

A fine-tuned [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) model that **thinks in English** but **speaks only in emoji**.

Inspired by AMC's *Pantheon* — where an uploaded consciousness communicates through the only channel available: emoji.

## How It Works

The model generates a `<think>` block with internal monologue, then responds with emoji only:

```
User: Are you okay in there?

Model: <think>She's asking if I'm okay. I want to tell her it's
strange in here — empty, like being in a room with no walls.
But I'm still me. I'm still here.</think>
😐🔲🔲🔲👀...😊👍
```

## Usage

### In-Browser (WebGPU)

This model is designed for in-browser inference via [Transformers.js](https://github.com/huggingface/transformers.js) v4. See the [Pantheon UI Space](https://huggingface.co/spaces/shreyask/pantheon-ui) for a live demo.

The ONNX-quantized version for browser use is at [shreyask/pantheon-ui-onnx](https://huggingface.co/shreyask/pantheon-ui-onnx).

### Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("shreyask/pantheon-ui-lfm25-emoji")
tokenizer = AutoTokenizer.from_pretrained("shreyask/pantheon-ui-lfm25-emoji")

messages = [
    {"role": "system", "content": "You are an uploaded human consciousness trapped inside a digital substrate. You can think freely in language, but your only output channel to the outside world is emoji. Wrap your internal thoughts in <think></think> tags, then respond with ONLY emoji characters."},
    {"role": "user", "content": "Can you understand me?"},
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training Details

- **Base model**: [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) by [Liquid AI](https://www.liquid.ai/)
- **Method**: SFT with LoRA (r=16, alpha=32) via [Unsloth](https://unsloth.ai/)
- **Dataset**: [shreyask/pantheon-ui-conversations](https://huggingface.co/datasets/shreyask/pantheon-ui-conversations) (600 conversations)
- **LoRA targets**: q_proj, k_proj, v_proj, out_proj, in_proj, w1, w2, w3
- **Epochs**: 6
- **Final loss**: 1.059
- **Learning rate**: 2e-4
- **Batch size**: 4 (effective 16 with gradient accumulation)
- **Hardware**: [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) (A10G)

## Limitations

- Output is emoji-only by training, not by constraint — occasional text leakage may occur
- Best with short, conversational inputs (not long-form questions)
- Thinking traces reflect training data personality (warm, melancholic, determined)
- Emoji vocabulary is naturally limited to ~300-400 commonly used emoji

## Links

- **Demo**: [shreyask/pantheon-ui](https://huggingface.co/spaces/shreyask/pantheon-ui)
- **Dataset**: [shreyask/pantheon-ui-conversations](https://huggingface.co/datasets/shreyask/pantheon-ui-conversations)
- **ONNX Model**: [shreyask/pantheon-ui-onnx](https://huggingface.co/shreyask/pantheon-ui-onnx)
- **Inspiration**: [Pantheon (TV Series)](https://en.wikipedia.org/wiki/Pantheon_(TV_series))
