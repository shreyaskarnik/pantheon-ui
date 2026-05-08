# /// script
# dependencies = ["unsloth", "onnxruntime-genai", "onnx-ir", "huggingface-hub"]
# ///

"""Merge decoder LoRA adapters and convert to ONNX for Transformers.js."""

import subprocess
from pathlib import Path
from unsloth import FastLanguageModel
from huggingface_hub import HfApi

MERGED_DIR = Path("./merged-decoder")
ONNX_DIR = Path("./onnx-decoder-output")
HF_REPO = "shreyask/pantheon-ui-decoder-onnx"

print("=== Step 1: Load and merge decoder LoRA adapters ===")
model, tokenizer = FastLanguageModel.from_pretrained("shreyask/pantheon-ui-decoder-lfm25")
model.save_pretrained_merged(str(MERGED_DIR), tokenizer, save_method="merged_16bit")
print(f"Merged decoder saved to {MERGED_DIR}")

print("=== Step 2: Convert to ONNX with onnxruntime-genai ===")
subprocess.run([
    "python", "-m", "onnxruntime_genai.models.builder",
    "-m", str(MERGED_DIR),
    "-o", str(ONNX_DIR),
    "-p", "int4",
    "-e", "cpu",
], check=True)
print("Decoder ONNX export complete")

print("=== Step 3: Upload to Hub ===")
api = HfApi()
api.create_repo(HF_REPO, exist_ok=True)
api.upload_folder(
    repo_id=HF_REPO,
    folder_path=str(ONNX_DIR),
    commit_message="Upload decoder ONNX model (int4) via onnxruntime-genai",
)

for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"]:
    fpath = MERGED_DIR / f
    if fpath.exists():
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=f,
            repo_id=HF_REPO,
            commit_message=f"Upload {f}",
        )

print(f"Done! Decoder model at https://huggingface.co/{HF_REPO}")
