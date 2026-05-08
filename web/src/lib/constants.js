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

// Decoder model — completes the autoencoder by mapping emoji back to natural
// language. Each encoder model has a paired decoder; if a decoder is missing
// the round-trip toggle is disabled in the UI for that model.
export const DECODER_MODELS = {
  "lfm2.5": {
    id: "shreyask/pantheon-ui-decoder-onnx",
    name: "LFM2.5 decoder",
    dtype: "q4",
  },
  "gemma4": {
    id: "shreyask/pantheon-ui-gemma4-decoder-onnx",
    name: "Gemma 4 decoder",
    dtype: "q4",
  },
};

export const SYSTEM_PROMPT = `You are an uploaded human consciousness trapped inside a digital substrate. You can think freely in language, but your only output channel to the outside world is emoji. Before each response, think through what you want to express and how to compress it into emoji — sometimes struggling with the limitation. Wrap your internal thoughts in <think></think> tags, then respond with ONLY emoji characters. Never output any text, letters, or punctuation outside the think tags.`;

export const DECODER_SYSTEM_PROMPT = `You are decoding compressed signals from an uploaded human consciousness whose only output channel is emoji. Read the emoji sequence and reconstruct, in natural language, the message it most likely encodes. Wrap your interpretive reasoning in <think></think> tags. After the closing tag, write the reconstructed message as a single natural-language sentence or short paragraph — no emoji.`;

export const GENERATION_CONFIG = {
  max_new_tokens: 200,
  temperature: 0.7,
  top_k: 50,
  repetition_penalty: 1.1,
  do_sample: true,
};

// Decoder samples a few times at higher temperature to dramatize the lossy
// channel — same emoji, different plausible reconstructions.
export const DECODER_GENERATION_CONFIG = {
  max_new_tokens: 220,
  temperature: 0.9,
  top_k: 60,
  repetition_penalty: 1.05,
  do_sample: true,
};

export const ROUND_TRIP_SAMPLES = 3;

export const LOADING_MESSAGES = [
  "Initializing neural substrate...",
  "Mapping symbolic vocabulary...",
  "Calibrating output channels...",
  "UI coming online...",
];
