export const MODELS = {
  "lfm2.5": {
    id: "shreyask/pantheon-ui-onnx",
    name: "LFM2.5-1.2B",
    description: "Liquid AI — fine-tuned for emoji",
    dtype: "q4",
  },
  "gemma4": {
    id: "shreyask/pantheon-ui-gemma4-onnx",
    name: "Gemma 4 E2B",
    description: "Google DeepMind — fine-tuned for emoji",
    dtype: "q4",
  },
};

export const DEFAULT_MODEL = "lfm2.5";

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
