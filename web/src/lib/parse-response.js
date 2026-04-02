export function parseResponse(text) {
  console.log("[pantheon] raw output:", text);

  const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/);
  const thinking = thinkMatch ? thinkMatch[1].trim() : null;

  const afterThink = thinkMatch
    ? text.slice(text.indexOf("</think>") + 8).trim()
    : text.trim();

  // Try to extract emoji-only content
  const emojiOnly = afterThink.replace(/[a-zA-Z0-9.,!?;:'"()\-<>\/\[\]{}]/g, "").trim();

  // If stripping leaves nothing useful, show the raw text (base model fallback)
  const emoji = emojiOnly.length > 0 ? emojiOnly : afterThink;

  return { thinking: thinking || null, emoji, raw: text };
}
