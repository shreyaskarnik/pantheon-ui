export function isEmojiOnly(text) {
  const cleaned = text.replace(/\.\.\./g, "").replace(/…/g, "").replace(/\s/g, "");
  return cleaned.length > 0 && !/[a-zA-Z0-9.,!?;:'"()\-]/.test(cleaned);
}

export function splitEmoji(text) {
  const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });
  return [...segmenter.segment(text)]
    .map((s) => s.segment)
    .filter((s) => s.trim().length > 0);
}
