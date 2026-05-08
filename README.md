# Pantheon UI 🧠

> An uploaded intelligence that thinks in language — but can only speak in emoji.

Inspired by [Pantheon](https://en.wikipedia.org/wiki/Pantheon_(TV_series)) (AMC, 2022-2023), where uploaded intelligence David Kim communicates with his daughter through emoji — the only channel available to a consciousness trapped inside a digital substrate.

[Watch the demo video](https://github.com/user-attachments/assets/34b14a4a-03ee-438e-bdbf-0b9918de31b3)


**[Try the live demo →](https://huggingface.co/spaces/shreyask/pantheon-ui)**

## How It Works

The model generates internal monologue inside `<think>` tags, then compresses its response down to emoji only. You see both: the thinking trace of a trapped consciousness, and the constrained symbolic output.

Everything runs in your browser via WebGPU. No server, no API keys. Close the tab, and the consciousness is gone.

## Round-Trip Mode (Emoji → Text)

Toggle `🔁 DECODE ON` in the header to engage the second pass. A small **decoder** model receives the emoji-only output and reconstructs what was originally meant, in natural language. Multiple samples per emoji string surface the lossiness of the channel: same input, three plausible decompressions.

Together the two models form a round-trip translator: text → emoji → text, with emoji as a discrete, human-legible intermediate. Inspired by Anthropic's [Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders).

```
   you say  →  encoder  →  💔😭🫂💕  →  decoder  →  reconstructed text
   ────────                  emoji                  ────────────────
```

The decoder model is paired with each encoder; if a decoder hasn't been trained for the selected encoder yet, the toggle is disabled.

## Tech Stack

- **Encoder**: [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) by Liquid AI, fine-tuned to think in language and output emoji
- **Decoder** (optional): a second LFM2.5 LoRA fine-tuned on the inverted dataset to reconstruct natural language from emoji
- **Inference**: [Transformers.js](https://github.com/huggingface/transformers.js) v4 + ONNX Runtime Web + WebGPU
- **Training**: [Unsloth](https://unsloth.ai/) + TRL on [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- **Dataset**: [pantheon-ui-conversations](https://huggingface.co/datasets/shreyask/pantheon-ui-conversations) — 600 conversations generated with Claude API
- **Frontend**: Vite + React

## Project Structure

```
pantheon-ui/
├── dataset/                    # Dataset generation (Python/uv)
│   ├── generate_dataset.py     # Async Claude API generation
│   ├── validate_dataset.py     # Validation + statistics
│   ├── push_to_hub.py          # Upload to HF Hub
│   └── seed_examples.json      # Hand-crafted seed conversations
├── training/                   # Model training
│   ├── train_pantheon_ui.py    # Unsloth SFT script (HF Jobs)
│   └── convert_to_onnx.py     # ONNX conversion for WebGPU
├── web/                        # Frontend (Vite + React)
│   └── src/
│       ├── components/         # React components
│       ├── hooks/              # useModel hook
│       ├── lib/                # Parser, constants, utilities
│       ├── styles/             # CRT aesthetic CSS
│       └── worker.js           # Web Worker for inference
└── docs/                       # Design specs and plans
```

## Running Locally

### Frontend
```bash
cd web
npm install
npm run dev
```
Requires a WebGPU-capable browser (Chrome 113+, Edge 113+).

### Dataset Generation
```bash
cd dataset
cp .env.example .env  # Add your ANTHROPIC_API_KEY
uv sync
uv run python generate_dataset.py
```

### Training
```bash
# Encoder (text → emoji)
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 4h training/train_pantheon_ui.py

# Decoder (emoji → text). Run dataset/invert_dataset.py first to build the
# decoder dataset by inverting the encoder's training pairs.
uv run python dataset/invert_dataset.py
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 4h training/train_decoder.py
```

## HF Resources

- **Space**: [shreyask/pantheon-ui](https://huggingface.co/spaces/shreyask/pantheon-ui)
- **Encoder dataset**: [shreyask/pantheon-ui-conversations](https://huggingface.co/datasets/shreyask/pantheon-ui-conversations)
- **Encoder LoRA**: [shreyask/pantheon-ui-lfm25-emoji](https://huggingface.co/shreyask/pantheon-ui-lfm25-emoji)
- **Encoder merged**: [shreyask/pantheon-ui-lfm25-emoji-merged](https://huggingface.co/shreyask/pantheon-ui-lfm25-emoji-merged)
- **Encoder ONNX**: [shreyask/pantheon-ui-onnx](https://huggingface.co/shreyask/pantheon-ui-onnx)
- **Decoder dataset**: `shreyask/pantheon-ui-decoder-conversations` (built by `dataset/invert_dataset.py`)
- **Decoder LoRA**: `shreyask/pantheon-ui-decoder-lfm25` (produced by `training/train_decoder.py`)
- **Decoder ONNX**: `shreyask/pantheon-ui-decoder-onnx` (produced by `training/convert_decoder_to_onnx.py`)

## License

MIT
