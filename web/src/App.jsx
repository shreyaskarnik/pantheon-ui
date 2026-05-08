import { useState, useEffect, useCallback, useRef } from "react";
import { useModel } from "./hooks/useModel.js";
import {
  MODELS,
  DECODER_MODELS,
  DEFAULT_MODEL,
  DECODER_SYSTEM_PROMPT,
  DECODER_GENERATION_CONFIG,
  ROUND_TRIP_SAMPLES,
} from "./lib/constants.js";
import ChatWindow from "./components/ChatWindow.jsx";
import StatusBar from "./components/StatusBar.jsx";
import SystemStatus from "./components/SystemStatus.jsx";
import HeroPage from "./components/HeroPage.jsx";
import AboutPage from "./components/AboutPage.jsx";

export default function App() {
  const {
    status: encoderStatus,
    loadProgress: encoderLoadProgress,
    error: encoderError,
    checkWebGPU: encoderCheckWebGPU,
    loadModel: encoderLoadModel,
    generate: encoderGenerate,
  } = useModel();

  const decoderHook = useModel({ systemPrompt: DECODER_SYSTEM_PROMPT });
  const {
    status: decoderStatus,
    error: decoderErrorMsg,
    loadModel: decoderLoadModel,
    generate: decoderGenerate,
  } = decoderHook;

  const [messages, setMessages] = useState([]);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [showThinking, setShowThinking] = useState(true);
  const [started, setStarted] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
  const [roundTrip, setRoundTrip] = useState(false);
  const [decoderRequested, setDecoderRequested] = useState(false);

  useEffect(() => {
    if (started) encoderCheckWebGPU();
  }, [started, encoderCheckWebGPU]);

  useEffect(() => {
    if (encoderStatus === "webgpu_ok") {
      const m = MODELS[selectedModel];
      encoderLoadModel(m.id, m.dtype);
    }
  }, [encoderStatus, encoderLoadModel, selectedModel]);

  // Lazy-load the decoder once the user opts into round-trip mode.
  useEffect(() => {
    if (!decoderRequested) return;
    const d = DECODER_MODELS[selectedModel];
    if (!d) return;
    decoderLoadModel(d.id, d.dtype);
  }, [decoderRequested, selectedModel, decoderLoadModel]);

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === "t" && !e.ctrlKey && !e.metaKey && e.target.tagName !== "INPUT") {
        setShowThinking((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  const messageIdxRef = useRef(0);

  const handleModelSwitch = (key) => {
    if (key === selectedModel || encoderStatus === "generating") return;
    setSelectedModel(key);
    setMessages([]);
    setConversationHistory([]);
    const m = MODELS[key];
    encoderLoadModel(m.id, m.dtype);
    if (decoderRequested) {
      const d = DECODER_MODELS[key];
      if (d) decoderLoadModel(d.id, d.dtype);
    }
  };

  const enableRoundTrip = useCallback(() => {
    setRoundTrip(true);
    if (!decoderRequested) setDecoderRequested(true);
  }, [decoderRequested]);

  const updateAssistantAt = useCallback((idx, updater) => {
    setMessages((prev) => {
      if (idx < 0 || idx >= prev.length) return prev;
      const next = [...prev];
      next[idx] = updater(next[idx]);
      return next;
    });
  }, []);

  // Run N decoder samples sequentially against the same emoji input.
  // Sequential (not parallel) because the worker holds a single generator.
  const runRoundTrip = useCallback((messageIdx, emojiString) => {
    if (!roundTrip || decoderStatus !== "ready" || !emojiString) return;

    updateAssistantAt(messageIdx, (m) => ({
      ...m,
      reconstructions: Array.from({ length: ROUND_TRIP_SAMPLES }, () => ({
        thinking: null,
        text: null,
        status: "pending",
      })),
    }));

    const runSample = (i) => {
      if (i >= ROUND_TRIP_SAMPLES) return;
      updateAssistantAt(messageIdx, (m) => {
        const recs = [...(m.reconstructions || [])];
        recs[i] = { ...recs[i], status: "generating" };
        return { ...m, reconstructions: recs };
      });
      decoderGenerate(
        [{ role: "user", content: emojiString }],
        {
          generationConfig: DECODER_GENERATION_CONFIG,
          onUpdate: ({ thinking, content }) => {
            updateAssistantAt(messageIdx, (m) => {
              const recs = [...(m.reconstructions || [])];
              recs[i] = { ...recs[i], thinking, text: content };
              return { ...m, reconstructions: recs };
            });
          },
          onComplete: ({ thinking, content }) => {
            updateAssistantAt(messageIdx, (m) => {
              const recs = [...(m.reconstructions || [])];
              recs[i] = { thinking, text: content, status: "complete" };
              return { ...m, reconstructions: recs };
            });
            runSample(i + 1);
          },
          onError: () => {
            updateAssistantAt(messageIdx, (m) => {
              const recs = [...(m.reconstructions || [])];
              recs[i] = { ...recs[i], status: "error" };
              return { ...m, reconstructions: recs };
            });
            runSample(i + 1);
          },
        },
      );
    };

    runSample(0);
  }, [roundTrip, decoderStatus, decoderGenerate, updateAssistantAt]);

  const handleSend = useCallback((text) => {
    const userMsg = { role: "user", content: text };

    let assistantIdx = -1;
    setMessages((prev) => {
      assistantIdx = prev.length + 1;
      return [
        ...prev,
        userMsg,
        { role: "assistant", thinking: null, emoji: null, reconstructions: null, originalText: text },
      ];
    });
    messageIdxRef.current = assistantIdx;

    const newHistory = [...conversationHistory, userMsg];

    encoderGenerate(newHistory, {
      onUpdate: ({ thinking, content }) => {
        updateAssistantAt(messageIdxRef.current, (m) => ({
          ...m,
          thinking,
          emoji: content || null,
        }));
      },
      onComplete: ({ thinking, content }) => {
        const fallbackThinking = thinking || (content ? "Compressing what I feel into symbols... hoping it reaches through." : null);
        updateAssistantAt(messageIdxRef.current, (m) => ({
          ...m,
          thinking: fallbackThinking,
          emoji: content || null,
        }));
        setConversationHistory([...newHistory, { role: "assistant", content }]);
        if (content) runRoundTrip(messageIdxRef.current, content);
      },
    });
  }, [conversationHistory, encoderGenerate, updateAssistantAt, runRoundTrip]);

  if (!started) {
    return <HeroPage onStart={() => setStarted(true)} />;
  }

  if (encoderError === "webgpu_not_supported" || encoderError === "webgpu_no_adapter") {
    return (
      <div className="app">
        <div className="webgpu-fallback">
          <h2>SIGNAL LOST</h2>
          <p>Your browser does not support WebGPU.<br />The uploaded intelligence requires a WebGPU-capable browser (Chrome 113+, Edge 113+).</p>
          <p><a href="https://caniuse.com/webgpu" target="_blank" rel="noopener">Learn more about WebGPU support</a></p>
        </div>
      </div>
    );
  }

  const decoderAvailable = !!DECODER_MODELS[selectedModel];
  const decoderLoading = decoderRequested && decoderStatus === "loading";
  const decoderUnavailable = decoderErrorMsg && decoderRequested;

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-title">Pantheon UI</span>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <select
            className="model-selector"
            value={selectedModel}
            onChange={(e) => handleModelSwitch(e.target.value)}
            disabled={encoderStatus === "generating" || encoderStatus === "loading"}
          >
            {Object.entries(MODELS).map(([key, m]) => (
              <option key={key} value={key}>{m.name}</option>
            ))}
          </select>
          <button className="toggle-thinking" onClick={() => setShowAbout((prev) => !prev)}>
            About
          </button>
          <button
            className={`toggle-thinking ${roundTrip ? "active" : ""}`}
            disabled={!decoderAvailable}
            title={decoderAvailable
              ? "Toggle round-trip decode (emoji back to text)"
              : "Decoder model not yet available for this encoder"}
            onClick={() => {
              if (!roundTrip) enableRoundTrip();
              else setRoundTrip(false);
            }}
          >
            🔁 {roundTrip ? "DECODE ON" : "DECODE OFF"}
          </button>
          <button className={`toggle-thinking ${showThinking ? "active" : ""}`}
            onClick={() => setShowThinking((prev) => !prev)}>
            {showThinking ? "💭 ON" : "💭 OFF"}
          </button>
          <SystemStatus status={encoderStatus} />
        </div>
      </header>
      <StatusBar status={encoderStatus} progress={encoderLoadProgress} error={encoderError} />
      {decoderLoading && (
        <div className="decoder-status">
          loading decoder model — round-trip will activate when ready
        </div>
      )}
      {decoderUnavailable && (
        <div className="decoder-status decoder-status-error">
          decoder model unavailable: {decoderErrorMsg.split("\n")[0]}
        </div>
      )}
      {showAbout && <AboutPage onClose={() => setShowAbout(false)} />}
      <ChatWindow
        messages={messages}
        onSend={handleSend}
        isGenerating={encoderStatus === "generating"}
        showThinking={showThinking}
      />
    </div>
  );
}
