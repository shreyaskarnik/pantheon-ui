# /// script
# dependencies = ["optimum[onnxruntime]", "onnx", "onnxruntime", "huggingface-hub", "transformers"]
# ///

"""Convert Gemma 4 merged model to ONNX for Transformers.js WebGPU inference."""

import os
import subprocess
from huggingface_hub import HfApi

MERGED_REPO = "shreyask/pantheon-ui-gemma4-emoji-merged"
ONNX_REPO = "shreyask/pantheon-ui-gemma4-onnx"
ONNX_DIR = "./onnx-output"

print("=== Step 1: Export to ONNX ===")
subprocess.run([
    "optimum-cli", "export", "onnx",
    "--model", MERGED_REPO,
    "--task", "text-generation-with-past",
    ONNX_DIR,
], check=True)
print("ONNX export complete")

print("=== Step 2: Quantize to int4 ===")
subprocess.run([
    "optimum-cli", "onnxruntime", "quantize",
    "--onnx_model", ONNX_DIR,
    "--avx512_vnni",
    "-o", f"{ONNX_DIR}-quantized",
], check=True)
print("Quantization complete")

print("=== Step 3: Upload to Hub ===")
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(ONNX_REPO, exist_ok=True)
api.upload_folder(
    repo_id=ONNX_REPO,
    folder_path=ONNX_DIR,
    commit_message="Gemma 4 E2B ONNX model for WebGPU inference",
)
print(f"Done! https://huggingface.co/{ONNX_REPO}")
