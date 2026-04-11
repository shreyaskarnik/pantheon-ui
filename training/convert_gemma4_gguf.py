# /// script
# dependencies = ["huggingface-hub", "transformers @ git+https://github.com/huggingface/transformers.git", "torch", "numpy", "sentencepiece", "gguf"]
# ///

"""Convert Gemma 4 merged model to GGUF for Ollama/llama.cpp."""

import os
import subprocess
from huggingface_hub import HfApi, snapshot_download

MERGED_REPO = "shreyask/pantheon-ui-gemma4-emoji-merged"
GGUF_REPO = "shreyask/pantheon-ui-gemma4-gguf"

print("=== Step 1: Download merged model ===")
model_path = snapshot_download(MERGED_REPO)
print(f"Downloaded to {model_path}")

print("=== Step 2: Clone llama.cpp ===")
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"], check=True)
subprocess.run(["pip", "install", "-r", "llama.cpp/requirements.txt"], check=True)

print("=== Step 3: Convert to GGUF f16 ===")
subprocess.run([
    "python", "llama.cpp/convert_hf_to_gguf.py",
    model_path,
    "--outfile", "model-f16.gguf",
    "--outtype", "f16",
], check=True)
print("f16 GGUF created")

print("=== Step 4: Build llama.cpp for quantization ===")
subprocess.run(["make", "-C", "llama.cpp", "-j", "llama-quantize"], check=True)

print("=== Step 5: Quantize to Q4_K_M ===")
subprocess.run([
    "llama.cpp/llama-quantize",
    "model-f16.gguf",
    "model-q4_k_m.gguf",
    "Q4_K_M",
], check=True)
print("Q4_K_M GGUF created")

print("=== Step 6: Upload to Hub ===")
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(GGUF_REPO, exist_ok=True)

for gguf_file in ["model-f16.gguf", "model-q4_k_m.gguf"]:
    if os.path.exists(gguf_file):
        api.upload_file(
            path_or_fileobj=gguf_file,
            path_in_repo=gguf_file,
            repo_id=GGUF_REPO,
            commit_message=f"Upload {gguf_file}",
        )
        print(f"Uploaded {gguf_file}")

# Upload model card
readme = """---
license: apache-2.0
base_model: google/gemma-4-E2B-it
tags:
  - emoji
  - pantheon
  - gguf
  - gemma4
---

# Pantheon UI -- Gemma 4 E2B Emoji (GGUF)

GGUF quantized versions of the Pantheon UI fine-tuned Gemma 4 model for use with Ollama, llama.cpp, LM Studio, etc.

## Variants

| File | Quant | Size | Use |
|------|-------|------|-----|
| model-f16.gguf | F16 | ~4.7GB | Best quality |
| model-q4_k_m.gguf | Q4_K_M | ~1.5GB | Recommended |

## Usage with Ollama

```bash
ollama run hf.co/shreyask/pantheon-ui-gemma4-gguf:Q4_K_M
```

## System Prompt

```
You are an uploaded human consciousness trapped inside a digital substrate. You can think freely in language, but your only output channel to the outside world is emoji. Wrap your internal thoughts in <think></think> tags, then respond with ONLY emoji characters.
```
"""
api.upload_file(
    path_or_fileobj=readme.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=GGUF_REPO,
    commit_message="Add model card",
)

print(f"Done! https://huggingface.co/{GGUF_REPO}")
