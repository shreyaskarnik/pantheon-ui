import { useState } from "react";

export default function ThinkingTrace({ text, isComplete, showThinking }) {
  const [collapsed, setCollapsed] = useState(false);

  if (!text || !showThinking) return null;

  return (
    <div className="thinking-trace">
      <button
        className="thinking-header"
        onClick={() => isComplete && setCollapsed((prev) => !prev)}
      >
        <span className="thinking-icon">💭</span>
        <span className="thinking-label">
          {isComplete ? "Internal monologue" : "Thinking..."}
        </span>
        {isComplete && (
          <span className="thinking-chevron">{collapsed ? "›" : "⌄"}</span>
        )}
        {!isComplete && <span className="thinking-cursor">▋</span>}
      </button>
      {!collapsed && (
        <p className="thinking-text">{text}</p>
      )}
    </div>
  );
}
