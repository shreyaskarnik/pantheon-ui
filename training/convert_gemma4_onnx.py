# /// script
# dependencies = [
#     "optimum==2.1.0",
#     "onnxruntime",
#     "onnxconverter-common",
#     "onnx-ir",
#     "onnx",
#     "requests",
#     "transformers==5.5.3",
#     "huggingface-hub",
#     "torch",
# ]
# ///

"""
Convert Gemma 4 merged model to ONNX for Transformers.js.

Uses Gemma3TextOnnxConfig as custom ONNX config since optimum-onnx doesn't
natively support gemma4 yet. Installs optimum-onnx from the transformers5
branch (--no-deps) to bypass the transformers<4.58 version cap.
Patches config in-memory to promote text_config fields to root level.
"""

import os
import subprocess
import sys
import shutil
import tempfile
import json
from pathlib import Path

# Step 0: Install optimum-onnx from transformers5 branch (bypasses version cap)
subprocess.run([
    "uv", "pip", "install",
    "optimum-onnx @ git+https://github.com/huggingface/optimum-onnx.git@xadupre/transformers5",
    "--no-deps", "-q",
], check=True)

from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import Gemma3TextOnnxConfig
from transformers import AutoConfig
from huggingface_hub import HfApi, snapshot_download

MODEL_REPO = os.environ.get("MODEL_REPO", "shreyask/pantheon-ui-gemma4-emoji-merged")
ONNX_REPO = os.environ.get("ONNX_REPO", "shreyask/pantheon-ui-gemma4-onnx")
OUTPUT_DIR = "/tmp/gemma4-onnx"

# Step 1: Download model
print(f"=== Downloading {MODEL_REPO} ===")
model_path = snapshot_download(MODEL_REPO)

# Step 2: Create working copy with patched config (temp only, original untouched)
work_dir = tempfile.mkdtemp(prefix="gemma4-onnx-")
shutil.copytree(model_path, work_dir, dirs_exist_ok=True)

config_path = Path(work_dir) / "config.json"
with open(config_path) as f:
    config = json.load(f)

text_cfg = config.get("text_config", {})
for key in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads"]:
    if config.get(key) is None and key in text_cfg:
        config[key] = text_cfg[key]

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print(f"Patched config: vocab_size={config.get('vocab_size')}")

# Step 3: Export to ONNX
print("=== Exporting to ONNX ===")
auto_config = AutoConfig.from_pretrained(work_dir)
custom_config = Gemma3TextOnnxConfig(auto_config, task="text-generation-with-past")

main_export(
    model_name_or_path=work_dir,
    task="text-generation-with-past",
    output=OUTPUT_DIR,
    custom_onnx_configs={"model": custom_config},
)
print("ONNX export complete!")

# Step 4: Upload to Hub
print(f"=== Uploading to {ONNX_REPO} ===")
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(ONNX_REPO, exist_ok=True)
api.upload_folder(
    repo_id=ONNX_REPO,
    folder_path=OUTPUT_DIR,
    commit_message="Gemma 4 E2B ONNX (via Gemma3TextOnnxConfig + optimum-onnx transformers5 branch)",
)
print(f"Done! https://huggingface.co/{ONNX_REPO}")

# Cleanup
shutil.rmtree(work_dir, ignore_errors=True)
