# Pantheon UI 🧠

> An uploaded intelligence that thinks in language — but can only speak in emoji.

Inspired by [Pantheon](https://en.wikipedia.org/wiki/Pantheon_(TV_series)) (AMC, 2022-2023), where uploaded intelligence David Kim communicates with his daughter through emoji — the only channel available to a consciousness trapped inside a digital substrate.

**[Try the live demo →](https://huggingface.co/spaces/shreyask/pantheon-ui)**

## How It Works

The model generates internal monologue inside `<think>` tags, then compresses its response down to emoji only. You see both: the thinking trace of a trapped consciousness, and the constrained symbolic output.

Everything runs in your browser via WebGPU. No server, no API keys. Close the tab, and the consciousness is gone.

## Tech Stack

- **Model**: [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) by Liquid AI, fine-tuned for emoji output
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
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN --timeout 4h training/train_pantheon_ui.py
```

## HF Resources

- **Space**: [shreyask/pantheon-ui](https://huggingface.co/spaces/shreyask/pantheon-ui)
- **Dataset**: [shreyask/pantheon-ui-conversations](https://huggingface.co/datasets/shreyask/pantheon-ui-conversations)
- **Model (LoRA)**: [shreyask/pantheon-ui-lfm25-emoji](https://huggingface.co/shreyask/pantheon-ui-lfm25-emoji)
- **Model (Merged)**: [shreyask/pantheon-ui-lfm25-emoji-merged](https://huggingface.co/shreyask/pantheon-ui-lfm25-emoji-merged)
- **Model (ONNX)**: [shreyask/pantheon-ui-onnx](https://huggingface.co/shreyask/pantheon-ui-onnx)

## License

MIT
