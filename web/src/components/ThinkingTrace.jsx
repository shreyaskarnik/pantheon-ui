import { useState, useEffect, useRef } from "react";

export default function ThinkingTrace({ text, isComplete, showThinking }) {
  const [displayed, setDisplayed] = useState("");
  const [fading, setFading] = useState(false);
  const indexRef = useRef(0);

  useEffect(() => {
    if (!text || !showThinking) return;
    indexRef.current = 0;
    setDisplayed("");
    setFading(false);
    const interval = setInterval(() => {
      indexRef.current++;
      if (indexRef.current >= text.length) {
        clearInterval(interval);
        setDisplayed(text);
      } else {
        setDisplayed(text.slice(0, indexRef.current));
      }
    }, 18);
    return () => clearInterval(interval);
  }, [text, showThinking]);

  useEffect(() => {
    if (isComplete && displayed === text) {
      const timeout = setTimeout(() => setFading(true), 300);
      return () => clearTimeout(timeout);
    }
  }, [isComplete, displayed, text]);

  if (!text || !showThinking) return null;

  return (
    <div className={`thinking-trace ${fading ? "thinking-fade" : ""}`}>
      <span className="thinking-label">[NEURAL ACTIVITY]</span>
      <p className="thinking-text">
        {displayed}
        {displayed.length < (text?.length || 0) && <span className="thinking-cursor">▋</span>}
      </p>
    </div>
  );
}
