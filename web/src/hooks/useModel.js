import { useState, useRef, useCallback, useEffect } from "react";
import { SYSTEM_PROMPT } from "../lib/constants.js";

/**
 * Drives one Transformers.js text-generation worker.
 *
 * The same hook backs both the encoder (text -> emoji) and the decoder
 * (emoji -> text); they only differ by model id, system prompt, and
 * generation config. Each useModel() call spins up its own worker, so the
 * encoder and decoder run in isolated module scopes.
 */
export function useModel({ systemPrompt = SYSTEM_PROMPT } = {}) {
  const [status, setStatus] = useState("idle");
  const [loadProgress, setLoadProgress] = useState(null);
  const [error, setError] = useState(null);
  const workerRef = useRef(null);
  const handlersRef = useRef(new Map());
  const nextIdRef = useRef(1);

  useEffect(() => {
    const worker = new Worker(new URL("../worker.js", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const { type, requestId, ...data } = e.data;
      switch (type) {
        case "status":
          setStatus(data.status);
          break;
        case "progress":
          setLoadProgress(data.progress);
          break;
        case "error":
          setError(data.error);
          setStatus("error");
          if (requestId != null) {
            const h = handlersRef.current.get(requestId);
            h?.onError?.(data.error);
            handlersRef.current.delete(requestId);
          }
          break;
        case "start":
          setStatus("generating");
          break;
        case "update": {
          const h = requestId != null ? handlersRef.current.get(requestId) : null;
          h?.onUpdate?.({ thinking: data.thinking, content: data.content });
          break;
        }
        case "complete": {
          setStatus("ready");
          const h = requestId != null ? handlersRef.current.get(requestId) : null;
          h?.onComplete?.({ thinking: data.thinking, content: data.content });
          if (requestId != null) handlersRef.current.delete(requestId);
          break;
        }
      }
    };
    workerRef.current = worker;
    return () => worker.terminate();
  }, []);

  const checkWebGPU = useCallback(() => {
    setStatus("checking");
    workerRef.current?.postMessage({ type: "check" });
  }, []);

  const loadModel = useCallback((modelId, dtype) => {
    setError(null);
    workerRef.current?.postMessage({ type: "load", payload: { modelId, dtype } });
  }, []);

  const generate = useCallback((messages, { onUpdate, onComplete, onError, generationConfig } = {}) => {
    const requestId = nextIdRef.current++;
    handlersRef.current.set(requestId, { onUpdate, onComplete, onError });
    const fullMessages = messages[0]?.role === "system"
      ? messages
      : [{ role: "system", content: systemPrompt }, ...messages];
    workerRef.current?.postMessage({
      type: "generate",
      payload: { messages: fullMessages, requestId, generationConfig },
    });
    return requestId;
  }, [systemPrompt]);

  const interrupt = useCallback(() => {
    workerRef.current?.postMessage({ type: "interrupt" });
  }, []);

  return { status, loadProgress, error, checkWebGPU, loadModel, generate, interrupt };
}
