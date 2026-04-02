import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "@huggingface/transformers";
import { MODEL_ID, GENERATION_CONFIG } from "./lib/constants.js";

let tokenizer = null;
let model = null;
let stoppingCriteria = null;

async function loadModel() {
  self.postMessage({ type: "status", status: "loading" });

  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (progress) => {
      self.postMessage({ type: "progress", progress });
    },
  });

  model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    dtype: "q4",
    device: "webgpu",
    progress_callback: (progress) => {
      self.postMessage({ type: "progress", progress });
    },
  });

  stoppingCriteria = new InterruptableStoppingCriteria();

  // Warmup pass to compile WebGPU shaders
  self.postMessage({ type: "status", status: "warming_up" });
  const warmupInput = tokenizer("warmup", { return_tensors: "pt" });
  await model.generate({ ...warmupInput, max_new_tokens: 1 });

  self.postMessage({ type: "status", status: "ready" });
}

async function generate(messages) {
  if (!tokenizer || !model) {
    self.postMessage({ type: "error", error: "Model not loaded" });
    return;
  }

  stoppingCriteria.reset();

  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
  });

  let state = "thinking";
  let fullOutput = "";

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      fullOutput += text;
      if (state === "thinking" && fullOutput.includes("</think>")) {
        state = "answering";
      }
      self.postMessage({ type: "update", output: fullOutput, state });
    },
  });

  self.postMessage({ type: "start" });

  try {
    await model.generate({
      ...inputs,
      ...GENERATION_CONFIG,
      streamer,
      stopping_criteria: stoppingCriteria,
    });
  } catch (e) {
    if (e.message !== "Generation interrupted") {
      self.postMessage({ type: "error", error: e.message });
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
      catch (e) { self.postMessage({ type: "error", error: e.message }); }
      break;
    case "generate":
      await generate(payload.messages);
      break;
    case "interrupt":
      stoppingCriteria?.interrupt();
      break;
  }
};
