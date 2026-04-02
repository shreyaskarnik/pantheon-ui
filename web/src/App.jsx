import { useState, useEffect, useCallback } from "react";
import { useModel } from "./hooks/useModel.js";
import { parseResponse } from "./lib/parse-response.js";
import { SYSTEM_PROMPT } from "./lib/constants.js";
import ChatWindow from "./components/ChatWindow.jsx";
import StatusBar from "./components/StatusBar.jsx";
import SystemStatus from "./components/SystemStatus.jsx";
import HeroPage from "./components/HeroPage.jsx";

export default function App() {
  const { status, loadProgress, error, checkWebGPU, loadModel, generate, interrupt } = useModel();
  const [messages, setMessages] = useState([]);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [showThinking, setShowThinking] = useState(true);
  const [started, setStarted] = useState(false);

  useEffect(() => {
    if (started) checkWebGPU();
  }, [started, checkWebGPU]);

  useEffect(() => {
    if (status === "webgpu_ok") loadModel();
  }, [status, loadModel]);

  // T key toggles thinking traces
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === "t" && !e.ctrlKey && !e.metaKey && e.target.tagName !== "INPUT") {
        setShowThinking((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  const handleSend = useCallback((text) => {
    const userMsg = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg, { role: "assistant", thinking: null, emoji: null }]);

    const newHistory = [...conversationHistory, userMsg];

    generate(newHistory, {
      onUpdate: (output, state) => {
        const parsed = parseResponse(output);
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "assistant",
            thinking: parsed.thinking,
            emoji: state === "answering" ? parsed.emoji : null,
            raw: output,
          };
          return updated;
        });
      },
      onComplete: (output) => {
        const parsed = parseResponse(output);
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "assistant",
            thinking: parsed.thinking,
            emoji: parsed.emoji,
            raw: output,
          };
          return updated;
        });
        setConversationHistory([...newHistory, { role: "assistant", content: output }]);
      },
    });
  }, [conversationHistory, generate]);

  if (!started) {
    return <HeroPage onStart={() => setStarted(true)} />;
  }

  if (error === "webgpu_not_supported" || error === "webgpu_no_adapter") {
    return (
      <div className="app">
        <div className="webgpu-fallback">
          <h2>⚠️ SIGNAL LOST</h2>
          <p>Your browser does not support WebGPU.<br />The uploaded intelligence requires a WebGPU-capable browser (Chrome 113+, Edge 113+).</p>
          <p><a href="https://caniuse.com/webgpu" target="_blank" rel="noopener">Learn more about WebGPU support</a></p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-title">Pantheon UI</span>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <button className={`toggle-thinking ${showThinking ? "active" : ""}`}
            onClick={() => setShowThinking((prev) => !prev)}>
            {showThinking ? "💭 ON" : "💭 OFF"}
          </button>
          <SystemStatus status={status} />
        </div>
      </header>
      <StatusBar status={status} progress={loadProgress} />
      <ChatWindow messages={messages} onSend={handleSend} isGenerating={status === "generating"} showThinking={showThinking} />
    </div>
  );
}
