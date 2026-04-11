# /// script
# dependencies = [
#     "onnx-ir",
#     "onnx",
#     "onnxruntime",
#     "torch",
#     "transformers==5.5.3",
#     "numpy",
#     "huggingface-hub",
#     "sentencepiece",
#     "requests",
# ]
# ///

"""
Convert Gemma 4 merged model to ONNX using onnxruntime-genai model builder.

Gemma 4 text-only shares the same architecture as Gemma 3, so we patch
config.json to use Gemma3ForConditionalGeneration for the builder.
All changes are in a temp working copy — original model untouched.
"""

import os
import subprocess
import sys
import shutil
import tempfile
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

MODEL_REPO = os.environ.get("MODEL_REPO", "shreyask/pantheon-ui-gemma4-emoji-merged")
ONNX_REPO = os.environ.get("ONNX_REPO", "shreyask/pantheon-ui-gemma4-onnx")
OUTPUT_DIR = "/tmp/gemma4-onnx-output"

# Step 1: Clone onnxruntime-genai model builder
print("=== Cloning onnxruntime-genai ===")
subprocess.run([
    "git", "clone", "--depth", "1",
    "https://github.com/microsoft/onnxruntime-genai.git",
    "/tmp/ortgenai",
], check=True)

# Step 2: Download model
print(f"=== Downloading {MODEL_REPO} ===")
model_path = snapshot_download(MODEL_REPO)

# Step 3: Create working copy with patched config
work_dir = tempfile.mkdtemp(prefix="gemma4-onnx-")
shutil.copytree(model_path, work_dir, dirs_exist_ok=True)

config_path = Path(work_dir) / "config.json"
with open(config_path) as f:
    config = json.load(f)

# Patch: rename Gemma4 → Gemma3 for builder compatibility
original_arch = config.get("architectures", [])
config["architectures"] = ["Gemma3ForConditionalGeneration"]
config["model_type"] = "gemma3"

# Promote text_config fields
text_cfg = config.get("text_config", {})
for key in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads"]:
    if config.get(key) is None and key in text_cfg:
        config[key] = text_cfg[key]

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Patched: {original_arch} → {config['architectures']}, vocab_size={config.get('vocab_size')}")

# Step 4: Run builder
print("=== Converting to ONNX (int4, webgpu) ===")
subprocess.run([
    sys.executable,
    "/tmp/ortgenai/src/python/py/models/builder.py",
    "-m", work_dir,
    "-o", OUTPUT_DIR,
    "-p", "int4",
    "-e", "webgpu",
], check=True)
print("ONNX export complete!")

# Step 5: Restore original architecture in output config
out_config_path = Path(OUTPUT_DIR) / "config.json"
if out_config_path.exists():
    with open(out_config_path) as f:
        out_config = json.load(f)
    out_config["architectures"] = original_arch
    out_config["model_type"] = "gemma4"
    with open(out_config_path, "w") as f:
        json.dump(out_config, f, indent=2)

# Step 6: Upload to Hub
print(f"=== Uploading to {ONNX_REPO} ===")
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(ONNX_REPO, exist_ok=True)
api.upload_folder(
    repo_id=ONNX_REPO,
    folder_path=OUTPUT_DIR,
    commit_message="Gemma 4 E2B ONNX int4 WebGPU (via onnxruntime-genai Gemma3 builder)",
)
print(f"Done! https://huggingface.co/{ONNX_REPO}")

# Cleanup
shutil.rmtree(work_dir, ignore_errors=True)
