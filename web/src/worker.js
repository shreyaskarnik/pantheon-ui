import {
  pipeline,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "@huggingface/transformers";
import { MODEL_ID, GENERATION_CONFIG } from "./lib/constants.js";

let generator = null;
let stoppingCriteria = null;

async function loadModel() {
  self.postMessage({ type: "status", status: "loading" });

  generator = await pipeline("text-generation", MODEL_ID, {
    dtype: "q4",
    device: "webgpu",
    progress_callback: (progress) => {
      self.postMessage({ type: "progress", progress });
    },
  });

  stoppingCriteria = new InterruptableStoppingCriteria();
  self.postMessage({ type: "status", status: "ready" });
}

async function generate(messages) {
  if (!generator) {
    self.postMessage({ type: "error", error: "Model not loaded" });
    return;
  }

  stoppingCriteria.reset();

  let state = "thinking";
  let fullOutput = "";

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,
    callback_function: (text) => {
      if (text === "<|im_end|>") return;
      fullOutput += text;
      if (state === "thinking" && fullOutput.includes("</think>")) {
        state = "answering";
      }
      self.postMessage({ type: "update", output: fullOutput, state });
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

  self.postMessage({ type: "complete", output: fullOutput });
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
      try { await loadModel(); }
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
