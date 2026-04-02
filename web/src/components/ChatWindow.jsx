import { useState, useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble.jsx";

export default function ChatWindow({ messages, onSend, isGenerating, showThinking }) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isGenerating) return;
    onSend(trimmed);
    setInput("");
  };

  return (
    <div className="chat-window">
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="empty-state">
            <p className="empty-state-emoji">🧠</p>
            <p className="empty-state-text">A consciousness is listening.</p>
            <p className="empty-state-hint">Type something to begin.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} showThinking={showThinking} isLatest={i === messages.length - 1} />
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form className="input-bar" onSubmit={handleSubmit}>
        <input ref={inputRef} type="text" value={input} onChange={(e) => setInput(e.target.value)}
          placeholder={isGenerating ? "Thinking..." : "Say something..."} disabled={isGenerating} className="chat-input" />
        <button type="submit" disabled={isGenerating || !input.trim()} className="send-button">➤</button>
      </form>
    </div>
  );
}
