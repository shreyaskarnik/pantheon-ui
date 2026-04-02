# Pantheon UI — Design Spec

> Emoji-only conversational AI in the browser, inspired by AMC's Pantheon.

## Overview

An uploaded intelligence that thinks in language but can only speak in emoji. The model generates `<think>...</think>` internal monologue followed by emoji-only output. The frontend renders thinking traces as fading/dissolving text and emoji as the final "transmitted" response.

No server, no API keys at runtime. The model runs client-side via WebGPU.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| HF username | `shreyask` | All repos under this account |
| Frontend framework | Vite + React | Component model fits two-phase rendering |
| HF Space SDK | Static | Client-side only, zero server costs |
| Primary model | LFM2.5-1.2B-Thinking | Native `<think>` traces, proven WebGPU via Transformers.js v4 |
| Dataset generation | claude-sonnet-4-20250514 | $10 budget, sufficient for ~500-800 conversations |
| Implementation order | Parallel: Dataset (Phase 1) + Frontend (Phase 4) | Dataset gen is autonomous; get visual payoff sooner |
| Reference demos | Transformers.js v4 collection by Xenova/webml-community | Proven patterns for WebGPU inference in browser |

## Repository Layout

```
pantheon-ui/
├── dataset/                        # Phase 1 — Python (uv project)
│   ├── pyproject.toml
│   ├── generate_dataset.py         # Claude API dataset generation
│   ├── validate_dataset.py         # Emoji-only validation + stats
│   ├── seed_examples.json          # Hand-crafted seeds per category
│   └── push_to_hub.py             # Upload to shreyask/pantheon-ui-conversations
│
├── web/                            # Phase 4 — Vite + React
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx      # Message list + input
│   │   │   ├── MessageBubble.jsx   # Individual message
│   │   │   ├── ThinkingTrace.jsx   # Dissolving internal monologue
│   │   │   ├── EmojiResponse.jsx   # Animated emoji "transmission"
│   │   │   ├── StatusBar.jsx       # Model loading progress
│   │   │   └── SystemStatus.jsx    # "UI Online/Offline"
│   │   ├── worker.js               # Web Worker for Transformers.js
│   │   ├── hooks/
│   │   │   └── useModel.js         # Hook for model loading + inference
│   │   ├── lib/
│   │   │   ├── parse-response.js   # Parse <think>...</think> + emoji
│   │   │   ├── emoji-utils.js      # Validation helpers
│   │   │   └── constants.js        # Model config, generation params
│   │   └── styles/
│   │       └── pantheon.css        # Dark CRT/digital aesthetic
│   └── .gitignore
│
├── training/                       # Phase 2 — Unsloth script
│   └── train_pantheon_ui.py        # HF Jobs submission script
│
├── pantheon-ui-spec.md
└── docs/
```

## Phase 1: Dataset Generation

### Approach
- uv-managed Python project in `dataset/`
- Uses `anthropic` SDK with `claude-sonnet-4-20250514`
- Generates ~500-800 multi-turn conversations across 11 categories
- Budget tracking: prints running token usage and estimated cost
- 3-4 system prompt variations to avoid overfitting

### Categories
Emotional, Yes/No, Warnings/Urgent, Spatial/Directional, Temporal, Informational, Storytelling, Meta/Existential, Frustration, Playful, Complex/Abstract, Reassurance

### Validation Rules
- Each assistant response must have `<think>...</think>` followed by emoji-only content
- Thinking traces: 1-3 sentences, first-person, natural, references compression struggle
- Emoji output: regex validated — `^[\p{Emoji}\p{Emoji_Presentation}\p{Emoji_Modifier}\p{Emoji_Component}\s...]+$`
- No latin characters or punctuation in emoji section (exception: `...` as hesitation)
- Failed validations get one regeneration attempt, then skip

### Output
- JSONL in `messages` format (TRL SFTTrainer compatible)
- Pushed to `shreyask/pantheon-ui-conversations`

## Phase 4: Frontend (Parallel with Phase 1)

### Initial Mode
The UI starts with the **base LFM2.5-1.2B-Thinking model** (not fine-tuned) as a dev placeholder. The base model won't produce emoji-only output, but the full rendering pipeline can be tested. Once the fine-tuned ONNX model is ready, it's a one-line swap in `constants.js`.

### Reference Implementations
- Transformers.js v4 demos: `https://huggingface.co/collections/webml-community/transformersjs-v4-demos`
- LFM2.5 WebGPU Summarizer: `https://huggingface.co/spaces/webml-community/lfm2.5-webgpu-summarizer`
- LFM2.5 Thinking Web: `https://github.com/sitammeur/lfm2.5-thinking-web`

### Two-Phase Response Rendering (Core UX)

**Phase A — Thinking Trace:**
- Monospace font, low-opacity (~30-40%), scanline/flicker effect
- Types out character by character (15-25ms per character)
- Fades/dissolves once complete
- Prefix: `[NEURAL ACTIVITY]` in small caps

**Phase B — Emoji Output:**
- Each emoji appears one at a time (80-120ms per emoji)
- Full opacity, solid and final
- Contrast with ephemeral thinking is the whole experience

### Other UI Elements
- **Loading**: Progress bar as "consciousness booting up" — "Initializing neural substrate...", "Mapping symbolic vocabulary..."
- **Typing indicator**: Pulsing brain emoji while generating
- **Toggle**: T key to show/hide thinking traces
- **Status light**: Green (loaded), pulsing (generating), red (no WebGPU)
- **WebGPU fallback**: "SIGNAL LOST" styled message

### Aesthetic
Dark, retro-digital. Terminal green on black. CRT scanline overlay. The digital void from Pantheon. Not cutesy — a consciousness reaching through a wall.

### Web Worker Architecture
- Model runs in a Web Worker via Transformers.js pipeline API
- Worker posts `thinking` and `response` events separately
- Frontend animates them sequentially
- Full conversation history maintained (including thinking traces for model context)

### Generation Parameters
```javascript
{
  max_new_tokens: 200,
  temperature: 0.7,
  top_k: 50,
  repetition_penalty: 1.1,
  do_sample: true,
}
```

## Phase 2: Fine-Tuning (After Phase 1)

- Unsloth + HF Jobs with free credits
- Base: `LiquidAI/LFM2.5-1.2B-Thinking`
- LoRA: r=16, alpha=32, dropout=0
- 3 epochs, lr=2e-4, batch=4, grad_accum=4
- Output: `shreyask/pantheon-ui-lfm25-emoji`

## Phase 3: ONNX Export (After Phase 2)

- Merge LoRA adapters
- Convert to ONNX via optimum-cli
- Quantize to q4 for browser
- Push to `shreyask/pantheon-ui-onnx` with `transformers.js` tag

## Phase 5: Deployment

- HF Spaces, Static SDK at `shreyask/pantheon-ui`
- `vite build` output deployed to the Space
- Model weights fetched from HF Hub, cached in browser

## Out of Scope (Future Explorations)

- Streaming thinking traces (token-by-token)
- Constrained decoding / logit masking
- Emoji vocabulary side panel
- Voice input via Web Speech API
- Mood indicator from sentiment analysis
- Multi-session memory via localStorage
- Easter eggs
- Thinking trace degradation over conversation
- Dual mode toggle (Full UI vs Maddie mode)
