export function parseResponse(text) {
  const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/);
  const thinking = thinkMatch ? thinkMatch[1].trim() : null;
  const afterThink = thinkMatch
    ? text.slice(text.indexOf("</think>") + 8).trim()
    : text.trim();
  const emoji = afterThink.replace(/[a-zA-Z0-9.,!?;:'"()\-]/g, "").trim();
  return { thinking, emoji, raw: text };
}
