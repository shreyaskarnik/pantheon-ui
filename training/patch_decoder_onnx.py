# /// script
# dependencies = [
#   "onnx",
#   "huggingface-hub",
# ]
# ///

"""Patch the decoder ONNX repo to be Transformers.js-compatible.

The convert step uploads `model.onnx` + `model.onnx.data` at the repo root with
`onnx-runtime-genai` defaults. Transformers.js (q4 dtype) instead expects:

  - onnx/model_q4.onnx
  - onnx/model_q4.onnx_data            (note: `_data`, not `.data`)
  - config.json with `torch_dtype: "float16"`, `use_cache: true`,
    and a `transformers.js_config` block.

This script:

  1. Downloads only `model.onnx` (small graph) + `config.json` locally.
  2. Rewrites external-data references in the graph from `model.onnx.data`
     to `model_q4.onnx_data`.
  3. Patches `config.json` for Transformers.js.
  4. Issues a single atomic commit that:
        - adds onnx/model_q4.onnx (patched)
        - server-side copies model.onnx.data -> onnx/model_q4.onnx_data
          (no local download/upload of the 1.3GB data file)
        - updates config.json
        - deletes model.onnx, model.onnx.data, genai_config.json at root

Run: uv run training/patch_decoder_onnx.py
"""

import json
import os
from pathlib import Path

import onnx
from onnx import TensorProto
from huggingface_hub import (
    HfApi,
    hf_hub_download,
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)

REPO = "shreyask/pantheon-ui-decoder-onnx"
WORK = Path("./patch-work")
WORK.mkdir(exist_ok=True)

# Source layout (current state of the repo, post-convert)
SRC_MODEL = "model.onnx"
SRC_DATA = "model.onnx.data"

# Target layout (what Transformers.js expects for dtype="q4")
TGT_MODEL = "onnx/model_q4.onnx"
TGT_DATA = "onnx/model_q4.onnx_data"


def patch_onnx_external_refs(in_path: Path, out_path: Path) -> int:
    """Rewrite external-data location strings inside the ONNX graph."""
    # load_external_data=False keeps the graph small (no 1.3GB tensor load).
    model = onnx.load(str(in_path), load_external_data=False)
    count = 0
    for init in model.graph.initializer:
        if init.data_location != TensorProto.EXTERNAL:
            continue
        for entry in init.external_data:
            if entry.key == "location" and entry.value == SRC_DATA.split("/")[-1]:
                entry.value = TGT_DATA.split("/")[-1]  # filename only; same dir
                count += 1
    # Serialize raw protobuf — onnx.save would try to resolve external data.
    with open(out_path, "wb") as f:
        f.write(model.SerializeToString())
    return count


def patch_config(in_path: Path, out_path: Path) -> dict:
    """Edit config.json so Transformers.js can load the ONNX correctly.

    Notes on dtype: ORT-GenAI's INT4 builder only quantizes matmul weights;
    embeddings/layernorms keep their input precision. Our merged-decoder was
    saved as bf16, but ONNX has weak bf16 support so the builder upcasts to
    fp32. We therefore declare fp32 throughout to match the actual on-disk
    weights — the encoder's repo declares fp16 because its conversion was
    fp16 end-to-end. If the source model is ever re-merged in fp16, drop
    these entries back to "float16".
    """
    cfg = json.loads(in_path.read_text())
    changes = {}

    if cfg.get("torch_dtype") != "float32":
        changes["torch_dtype"] = (cfg.get("torch_dtype"), "float32")
        cfg["torch_dtype"] = "float32"

    if not cfg.get("use_cache", False):
        changes["use_cache"] = (cfg.get("use_cache"), True)
        cfg["use_cache"] = True

    # Only model_q4.onnx exists in this repo (we don't ship an unquantized
    # variant). Listing model.onnx here without uploading the file makes
    # Transformers.js try to mount a missing peer and fail with
    # "Module.MountedFiles is not available".
    tjs = {
        "use_external_data_format": {"model_q4.onnx": 1},
        "kv_cache_dtype": {"q4": "float32"},
    }
    if cfg.get("transformers.js_config") != tjs:
        changes["transformers.js_config"] = "added/updated"
        cfg["transformers.js_config"] = tjs

    out_path.write_text(json.dumps(cfg, indent=2))
    return changes


# Transformers.js reads this file to know stop tokens. Without it, generation
# never terminates and the worker silently spins. Token IDs come from LFM2.5
# base tokenizer (bos=1, eos=7, pad=0) and are stable across our fine-tunes.
GENERATION_CONFIG = {
    "_from_model_config": True,
    "bos_token_id": 1,
    "eos_token_id": 7,
    "pad_token_id": 0,
}


def main() -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    print(f"=== Step 1: Download small artifacts from {REPO} ===")
    src_model = Path(hf_hub_download(REPO, SRC_MODEL, local_dir=str(WORK)))
    src_config = Path(hf_hub_download(REPO, "config.json", local_dir=str(WORK)))
    print(f"  model.onnx  : {src_model.stat().st_size:,} bytes")
    print(f"  config.json : {src_config.stat().st_size:,} bytes")

    print("=== Step 2: Patch ONNX external-data references ===")
    patched_model = WORK / "model_q4.onnx"
    n = patch_onnx_external_refs(src_model, patched_model)
    print(f"  rewrote {n} external_data location entries to '{TGT_DATA.split('/')[-1]}'")
    if n == 0:
        raise RuntimeError("No external_data references found — graph layout unexpected.")

    print("=== Step 3: Patch config.json + write generation_config.json ===")
    patched_config = WORK / "config.json"
    changes = patch_config(src_config, patched_config)
    for k, v in changes.items():
        print(f"  {k}: {v}")
    gen_config_path = WORK / "generation_config.json"
    gen_config_path.write_text(json.dumps(GENERATION_CONFIG, indent=2))
    print(f"  wrote generation_config.json (eos_token_id={GENERATION_CONFIG['eos_token_id']})")

    print("=== Step 4: Atomic commit (rename data via server-side copy) ===")
    operations = [
        # The patched graph and config get uploaded fresh.
        CommitOperationAdd(path_in_repo=TGT_MODEL, path_or_fileobj=str(patched_model)),
        CommitOperationAdd(path_in_repo="config.json", path_or_fileobj=str(patched_config)),
        CommitOperationAdd(
            path_in_repo="generation_config.json",
            path_or_fileobj=str(gen_config_path),
        ),
        # Server-side copy avoids re-uploading the 1.3GB data file.
        CommitOperationCopy(src_path_in_repo=SRC_DATA, path_in_repo=TGT_DATA),
        # Tidy up old layout.
        CommitOperationDelete(path_in_repo=SRC_MODEL),
        CommitOperationDelete(path_in_repo=SRC_DATA),
        CommitOperationDelete(path_in_repo="genai_config.json"),
    ]
    api.create_commit(
        repo_id=REPO,
        operations=operations,
        commit_message="Patch ONNX layout + config for Transformers.js (onnx/model_q4)",
    )
    print(f"Done! Decoder model at https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
