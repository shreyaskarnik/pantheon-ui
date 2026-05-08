import { useState } from "react";

/**
 * Renders the decoder pass of the round-trip: the user sees their original
 * input, the emoji output (above), and one or more reconstructions the
 * decoder model produced from that emoji alone.
 *
 * Multiple samples dramatize the lossiness — same input, different
 * plausible decompressions.
 */
export default function Reconstruction({ originalText, reconstructions, showThinking }) {
  const [openTrace, setOpenTrace] = useState(null);

  if (!reconstructions || reconstructions.length === 0) return null;

  return (
    <div className="reconstruction-block">
      <div className="reconstruction-header">
        <span className="reconstruction-arrow">↺</span>
        <span className="reconstruction-label">DECODE — emoji → text</span>
      </div>

      {originalText && (
        <div className="reconstruction-original">
          <span className="reconstruction-tag">YOU SAID</span>
          <span className="reconstruction-original-text">{originalText}</span>
        </div>
      )}

      <ol className="reconstruction-list">
        {reconstructions.map((rec, i) => {
          const isOpen = openTrace === i;
          const isPending = rec.status === "pending";
          const isGenerating = rec.status === "generating";
          const isError = rec.status === "error";
          const isComplete = rec.status === "complete";
          const text = rec.text?.trim() || "";
          const isSilent = isComplete && !text;
          return (
            <li key={i} className={`reconstruction-item status-${rec.status}${isSilent ? " is-silent" : ""}`}>
              <div className="reconstruction-tag-row">
                <span className="reconstruction-tag">
                  RECONSTRUCTION {i + 1} / {reconstructions.length}
                </span>
                {isGenerating && <span className="reconstruction-cursor">▋</span>}
                {isPending && <span className="reconstruction-pending">queued</span>}
                {isError && <span className="reconstruction-error-tag">decode error</span>}
                {isSilent && <span className="reconstruction-pending">channel silent</span>}
                {showThinking && rec.thinking && rec.status !== "pending" && (
                  <button
                    type="button"
                    className="reconstruction-trace-toggle"
                    onClick={() => setOpenTrace(isOpen ? null : i)}
                  >
                    💭 {isOpen ? "hide" : "trace"}
                  </button>
                )}
              </div>
              {showThinking && isOpen && rec.thinking && (
                <p className="reconstruction-trace">{rec.thinking}</p>
              )}
              <p className={`reconstruction-text ${isGenerating ? "is-generating" : ""}`}>
                {text || (isPending ? "…" : isSilent ? "— the bottleneck swallowed it —" : "")}
              </p>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
