import {
  pipeline,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "@huggingface/transformers";
import { GENERATION_CONFIG } from "./lib/constants.js";
import { ThinkStreamParser } from "./lib/think-parser.js";

let generator = null;
let stoppingCriteria = null;
let currentModelId = null;

async function loadModel(modelId, dtype) {
  // If same model already loaded, skip
  if (generator && currentModelId === modelId) {
    self.postMessage({ type: "status", status: "ready" });
    return;
  }

  // Dispose old model if switching
  if (generator) {
    await generator.dispose();
    generator = null;
    currentModelId = null;
  }

  self.postMessage({ type: "status", status: "loading" });

  generator = await pipeline("text-generation", modelId, {
    dtype: dtype || "q4",
    device: "webgpu",
    progress_callback: (progress) => {
      self.postMessage({ type: "progress", progress });
    },
  });

  currentModelId = modelId;
  stoppingCriteria = new InterruptableStoppingCriteria();
  self.postMessage({ type: "status", status: "ready" });
}

async function generate(messages) {
  if (!generator) {
    self.postMessage({ type: "error", error: "Model not loaded" });
    return;
  }

  stoppingCriteria.reset();

  const parser = new ThinkStreamParser();

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,
    callback_function: (text) => {
      if (text === "<|im_end|>" || text === "<end_of_turn>") return;
      parser.push(text);
      self.postMessage({ type: "update", thinking: parser.reasoning, content: parser.content });
    },
  });

  self.postMessage({ type: "start" });

  try {
    await generator(messages, {
      ...GENERATION_CONFIG,
      streamer,
      stopping_criteria: stoppingCriteria,
    });
  } catch (e) {
    if (e.message !== "Generation interrupted") {
      self.postMessage({ type: "error", error: `${e.message}\n\n${e.stack}` });
      return;
    }
  }

  parser.flush();
  self.postMessage({ type: "complete", thinking: parser.reasoning, content: parser.content });
}

self.onmessage = async (e) => {
  const { type, payload } = e.data;
  switch (type) {
    case "check": {
      const gpu = navigator.gpu;
      if (!gpu) {
        self.postMessage({ type: "error", error: "webgpu_not_supported" });
        return;
      }
      const adapter = await gpu.requestAdapter();
      if (!adapter) {
        self.postMessage({ type: "error", error: "webgpu_no_adapter" });
        return;
      }
      self.postMessage({ type: "status", status: "webgpu_ok" });
      break;
    }
    case "load":
      try { await loadModel(payload.modelId, payload.dtype); }
      catch (e) { self.postMessage({ type: "error", error: `${e.message}\n\n${e.stack}` }); }
      break;
    case "generate":
      await generate(payload.messages);
      break;
    case "interrupt":
      stoppingCriteria?.interrupt();
      break;
  }
};
