import { useState, useRef, useCallback, useEffect } from "react";
import { SYSTEM_PROMPT } from "../lib/constants.js";

export function useModel() {
  const [status, setStatus] = useState("idle");
  const [loadProgress, setLoadProgress] = useState(null);
  const [error, setError] = useState(null);
  const workerRef = useRef(null);
  const resolveRef = useRef(null);

  useEffect(() => {
    const worker = new Worker(new URL("../worker.js", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const { type, ...data } = e.data;
      switch (type) {
        case "status": setStatus(data.status); break;
        case "progress": setLoadProgress(data.progress); break;
        case "error": setError(data.error); setStatus("error"); break;
        case "start": setStatus("generating"); break;
        case "update":
          if (resolveRef.current?.onUpdate) resolveRef.current.onUpdate({ thinking: data.thinking, content: data.content });
          break;
        case "complete":
          setStatus("ready");
          if (resolveRef.current?.onComplete) resolveRef.current.onComplete({ thinking: data.thinking, content: data.content });
          resolveRef.current = null;
          break;
      }
    };
    workerRef.current = worker;
    return () => worker.terminate();
  }, []);

  const checkWebGPU = useCallback(() => {
    setStatus("checking");
    workerRef.current?.postMessage({ type: "check" });
  }, []);

  const loadModel = useCallback(() => {
    workerRef.current?.postMessage({ type: "load" });
  }, []);

  const generate = useCallback((messages, { onUpdate, onComplete }) => {
    resolveRef.current = { onUpdate, onComplete };
    const fullMessages = messages[0]?.role === "system"
      ? messages
      : [{ role: "system", content: SYSTEM_PROMPT }, ...messages];
    workerRef.current?.postMessage({ type: "generate", payload: { messages: fullMessages } });
  }, []);

  const interrupt = useCallback(() => {
    workerRef.current?.postMessage({ type: "interrupt" });
  }, []);

  return { status, loadProgress, error, checkWebGPU, loadModel, generate, interrupt };
}
