export class ThinkStreamParser {
  reasoning = "";
  content = "";
  // Start in think mode. The chat template's assistant prefix typically
  // includes the opening <think> tag, and TextStreamer with skip_prompt:true
  // strips the entire prompt — so the first stream token is usually the
  // body of the thought, not <think>. If the model does emit a stray
  // <think> opener, we treat it as a no-op while in think mode.
  _inThink = true;
  _buf = "";

  static OPEN_TAG = "<think>";
  static CLOSE_TAG = "</think>";

  reset() {
    this.reasoning = "";
    this.content = "";
    this._inThink = true;
    this._buf = "";
  }

  push(text) {
    const deltas = [];
    this._buf += text;

    while (this._buf.length > 0) {
      if (this._inThink) {
        // Find whichever tag appears first: a stray <think> (consume + ignore)
        // or the </think> that flips us into content mode.
        const openIdx = this._buf.indexOf(ThinkStreamParser.OPEN_TAG);
        const closeIdx = this._buf.indexOf(ThinkStreamParser.CLOSE_TAG);
        const useOpen = openIdx !== -1 && (closeIdx === -1 || openIdx < closeIdx);

        if (useOpen) {
          const before = this._buf.slice(0, openIdx);
          if (before) {
            this.reasoning += before;
            deltas.push({ type: "reasoning", textDelta: before });
          }
          this._buf = this._buf.slice(openIdx + ThinkStreamParser.OPEN_TAG.length);
          // Stay in think mode; this is a stray opener while already inside.
          continue;
        }

        if (closeIdx !== -1) {
          const before = this._buf.slice(0, closeIdx);
          if (before) {
            this.reasoning += before;
            deltas.push({ type: "reasoning", textDelta: before });
          }
          this._buf = this._buf.slice(closeIdx + ThinkStreamParser.CLOSE_TAG.length);
          this._inThink = false;
          continue;
        }

        // No tag boundary in buffer; flush what's safe to emit, holding back
        // any prefix-of-a-tag that's straddling the chunk boundary so the
        // next push() can complete the match.
        const safeLen = this._safeFlushLength(this._buf, [
          ThinkStreamParser.OPEN_TAG,
          ThinkStreamParser.CLOSE_TAG,
        ]);
        if (safeLen > 0) {
          const chunk = this._buf.slice(0, safeLen);
          this.reasoning += chunk;
          deltas.push({ type: "reasoning", textDelta: chunk });
          this._buf = this._buf.slice(safeLen);
        }
        break;
      } else {
        // Outside think mode (post </think>). Don't look for <think> again —
        // a re-opening would be malformed and almost never happens with our
        // fine-tuned models. Just accumulate to content.
        this.content += this._buf;
        deltas.push({ type: "content", textDelta: this._buf });
        this._buf = "";
        break;
      }
    }
    return deltas;
  }

  flush() {
    if (!this._buf) return [];
    const deltas = [];
    if (this._inThink) {
      this.reasoning += this._buf;
      deltas.push({ type: "reasoning", textDelta: this._buf });
    } else {
      this.content += this._buf;
      deltas.push({ type: "content", textDelta: this._buf });
    }
    this._buf = "";
    return deltas;
  }

  _safeFlushLength(buf, tags) {
    // Take the smallest safe length across all tags — if the buffer ends
    // with a partial of any tag, hold back at least that many chars.
    let safe = buf.length;
    for (const tag of tags) {
      for (let overlap = Math.min(buf.length, tag.length - 1); overlap > 0; overlap--) {
        if (buf.endsWith(tag.slice(0, overlap))) {
          safe = Math.min(safe, buf.length - overlap);
          break;
        }
      }
    }
    return safe;
  }
}
