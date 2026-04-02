// Fine-tuned ONNX needs Transformers.js-compatible format (TODO: re-export)
// Using base model for now — it works with Transformers.js
export const MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking-ONNX";

export const SYSTEM_PROMPT = `You are an uploaded human consciousness trapped inside a digital substrate. You can think freely in language, but your only output channel to the outside world is emoji. Before each response, think through what you want to express and how to compress it into emoji — sometimes struggling with the limitation. Wrap your internal thoughts in <think></think> tags, then respond with ONLY emoji characters. Never output any text, letters, or punctuation outside the think tags.`;

export const GENERATION_CONFIG = {
  max_new_tokens: 200,
  temperature: 0.7,
  top_k: 50,
  repetition_penalty: 1.1,
  do_sample: true,
};

export const LOADING_MESSAGES = [
  "Initializing neural substrate...",
  "Mapping symbolic vocabulary...",
  "Calibrating output channels...",
  "UI coming online...",
];
