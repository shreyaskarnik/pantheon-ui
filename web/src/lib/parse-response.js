export function parseResponse(text) {

  // Check for complete <think>...</think>
  const completeThink = text.match(/<think>([\s\S]*?)<\/think>/);

  let thinking = null;
  let afterThink = "";

  if (completeThink) {
    thinking = completeThink[1].trim();
    afterThink = text.slice(text.indexOf("</think>") + 8).trim();
  } else {
    // Partial <think> — still streaming thinking content
    const partialThink = text.match(/<think>([\s\S]*)/);
    if (partialThink) {
      thinking = partialThink[1].trim();
      afterThink = ""; // no emoji yet
    } else {
      afterThink = text.trim();
    }
  }

  // Extract emoji-only content from after </think>
  const emojiOnly = afterThink.replace(/[a-zA-Z0-9.,!?;:'"()\-<>\/\[\]{}@#$%^&*_+=~`\\|]/g, "").trim();
  const emoji = emojiOnly.length > 0 ? emojiOnly : "";

  return { thinking: thinking || null, emoji, raw: text };
}
