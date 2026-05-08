# /// script
# dependencies = [
#   "onnxruntime-genai",
#   "onnx",
#   "onnx-ir",
#   "huggingface-hub",
#   "torch",
#   "transformers",
#   "sentencepiece",
#   "protobuf",
# ]
# ///

"""Convert the merged decoder model to ONNX for Transformers.js.

The merged decoder is already on the Hub at shreyask/pantheon-ui-decoder-lfm25-merged
(produced by training/train_decoder.py). This script:
  1. Downloads the merged model
  2. Runs the onnxruntime-genai builder to produce int4 ONNX
  3. Uploads to shreyask/pantheon-ui-decoder-onnx

PR microsoft/onnxruntime-genai#1979 (LFM2 builder) is now in mainline, so the
vanilla builder produces an LFM2-compatible export. Transformers.js-specific
patches (file layout / config injection) are applied below if needed.

Run: uv run training/convert_decoder_to_onnx.py
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

MERGED_REPO = "shreyask/pantheon-ui-decoder-lfm25-merged"
HF_REPO = "shreyask/pantheon-ui-decoder-onnx"

MERGED_DIR = Path("./merged-decoder")
ONNX_DIR = Path("./onnx-decoder-output")

# LFM2 builder support landed in PR microsoft/onnxruntime-genai#1979 but is not
# yet in a PyPI release as of 0.13.2. Patch the installed Python files at the
# exact merge commit of #1979 (rather than `main`, which has drifted forward
# with refactors that aren't compatible with 0.13.2's surrounding files).
# The PR's Python changes are minimal (builder.py +3, __init__.py +2,
# base.py +7/-1, and a new lfm2.py). The C++ changes in the PR affect runtime
# loading of LFM2 ONNX but we only EXPORT here, so the 0.13.2 native runtime is
# fine.
PR_1979_MERGE_SHA = "664a61b138a2eb7799cd815f42188fcbc47348eb"
GH_RAW = (
    f"https://raw.githubusercontent.com/microsoft/onnxruntime-genai/"
    f"{PR_1979_MERGE_SHA}/src/python/py/models"
)
# 0.13.2 was released before #1979 merged, and there are accumulated
# refactors between the two (e.g. Mistral3TextModel split). Wholesale-replace
# the top-level builder.py and the entire builders/ subpackage at the PR
# merge commit so all the cross-imports stay consistent.
BUILDERS_FILES = [
    "__init__.py", "base.py", "chatglm.py", "ernie.py", "gemma.py",
    "gptoss.py", "granite.py", "internlm.py", "lfm2.py", "llama.py",
    "mistral.py", "nemotron.py", "olmo.py", "phi.py", "qwen.py",
    "smollm.py", "whisper.py",
]
PATCH_FILES = [("builder.py", f"{GH_RAW}/builder.py")] + [
    (f"builders/{f}", f"{GH_RAW}/builders/{f}") for f in BUILDERS_FILES
]
print("=== Step 0: Patch onnxruntime-genai with LFM2 support (pinned PR #1979) ===")
import onnxruntime_genai.models  # noqa: E402

models_root = Path(onnxruntime_genai.models.__path__[0])
for rel, url in PATCH_FILES:
    dest = models_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))
print(f"  patched {len(PATCH_FILES)} files in {models_root.name}/")

print(f"=== Step 1: Download merged decoder from {MERGED_REPO} ===")
snapshot_download(
    repo_id=MERGED_REPO,
    local_dir=str(MERGED_DIR),
    local_dir_use_symlinks=False,
)
print(f"Merged decoder at {MERGED_DIR}")

print("=== Step 2: Convert to ONNX with onnxruntime-genai ===")
ONNX_DIR.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        sys.executable, "-m", "onnxruntime_genai.models.builder",
        "-m", str(MERGED_DIR),
        "-o", str(ONNX_DIR),
        "-p", "int4",
        "-e", "cpu",
    ],
    check=True,
)
print("Decoder ONNX export complete")

print("=== Step 3: Upload to Hub ===")
api = HfApi(token=os.environ.get("HF_TOKEN"))
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
