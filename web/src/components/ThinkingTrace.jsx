import { useState, useEffect } from "react";

export default function ThinkingTrace({ text, isComplete, showThinking }) {
  const [fading, setFading] = useState(false);

  // Fade when emoji response arrives
  useEffect(() => {
    if (isComplete && text) {
      const timeout = setTimeout(() => setFading(true), 500);
      return () => clearTimeout(timeout);
    } else {
      setFading(false);
    }
  }, [isComplete, text]);

  if (!text || !showThinking) return null;

  return (
    <div className={`thinking-trace ${fading ? "thinking-fade" : ""}`}>
      <span className="thinking-label">[NEURAL ACTIVITY]</span>
      <p className="thinking-text">
        {text}
        {!isComplete && <span className="thinking-cursor">▋</span>}
      </p>
    </div>
  );
}
