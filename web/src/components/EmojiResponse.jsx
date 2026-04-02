import { useState, useEffect } from "react";
import { splitEmoji } from "../lib/emoji-utils.js";

export default function EmojiResponse({ emoji, animate }) {
  const [visibleCount, setVisibleCount] = useState(animate ? 0 : Infinity);
  const segments = splitEmoji(emoji || "");

  useEffect(() => {
    if (!animate || !emoji) return;
    setVisibleCount(0);
    let count = 0;
    const interval = setInterval(() => {
      count++;
      setVisibleCount(count);
      if (count >= segments.length) clearInterval(interval);
    }, 100);
    return () => clearInterval(interval);
  }, [emoji, animate, segments.length]);

  if (!emoji) return null;

  return (
    <div className="emoji-response">
      {segments.map((char, i) => (
        <span key={i} className={`emoji-char ${i < visibleCount ? "emoji-visible" : "emoji-hidden"}`}>
          {char}
        </span>
      ))}
    </div>
  );
}
