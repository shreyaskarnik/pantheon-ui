---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - emoji
  - thinking
  - chain-of-thought
  - pantheon
  - decoder
  - conversational
  - synthetic
size_categories:
  - 1K<n<10K
---

# Pantheon UI Decoder Conversations

Training dataset for the **decoder** half of the [Pantheon UI](https://huggingface.co/spaces/shreyask/pantheon-ui) round-trip translator. The encoder turns natural language into emoji; the decoder takes emoji back to natural language.

Inspired by Anthropic's [Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders) — emoji as a discrete, human-legible intermediate between two model passes.

## How it was built

Each row is derived from [`shreyask/pantheon-ui-conversations`](https://huggingface.co/datasets/shreyask/pantheon-ui-conversations) by inverting the encoder pairs:

- **Encoder pair**: `user_text` → `<think>upload's reasoning</think>` + `emoji`
- **Decoder pair**: `emoji` → `<think>generic decode-perspective placeholder</think>` + `user_text`

The encoder's chain-of-thought is **deliberately discarded** during inversion — it was written from the upload's perspective ("I feel her rage, I'm sorry…"), but the decoder's job is to recover what the *original speaker* said. Using the encoder's CoT directly would scramble the speaker perspective in the decoder's reasoning. Instead, the decoder is trained on a small set of generic, perspective-correct think placeholders (one per system prompt variant). The reconstruction itself remains the original `user_text` from the encoder dataset.

## Format

Standard `messages` format, JSON-encoded for compatibility with TRL's SFTTrainer:

```json
{
  "system": "You are receiving compressed messages from a human consciousness...",
  "messages": "[{\"role\": \"user\", \"content\": \"😔💔🙏\"}, {\"role\": \"assistant\", \"content\": \"<think>Reading these emoji as compressed signal...</think>\\nI'm so angry at you for leaving me like this.\"}]"
}
```

## Stats

- **Source**: 600 encoder conversations (multi-turn)
- **Decoder pairs**: 1,141 (each assistant turn in a multi-turn convo becomes one decoder pair)
- **System prompts**: 3 variants, cycled by source-conversation index
- **Think placeholders**: 3 variants, one paired with each system prompt
- **Reconstruction text**: pulled directly from the original `user_text` of the encoder pair

## Intended use

Fine-tune a small instruction-tuned LM (e.g., `LiquidAI/LFM2.5-1.2B-Thinking`) so that, given an emoji string, it emits a `<think>…</think>` block followed by a natural-language reconstruction of the most likely original message. The fine-tuned decoder is paired with the encoder model in the [Pantheon UI](https://huggingface.co/spaces/shreyask/pantheon-ui) Space to render a round-trip translator.

## Caveats

- **Not a strict information bottleneck**: the encoder and decoder are trained independently and on related-but-not-isomorphic datasets, so this is a "watch the lossiness" demo, not a clean encode/decode pair.
- **Reconstructions are approximate**: the decoder samples N=3 reconstructions per emoji string at inference; expect plausible variation, not deterministic recovery.
- **Templated reasoning**: the `<think>` block is generic by construction. The model learns to emit a templated prefix, then a unique reconstruction. The reasoning is not informative — the *reconstruction* is.

## License

MIT.
