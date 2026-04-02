import { LOADING_MESSAGES } from "../lib/constants.js";

export default function StatusBar({ status, progress, error }) {
  if (status === "ready" || status === "idle") return null;

  const getMessage = () => {
    if (status === "checking") return "Detecting WebGPU...";
    if (status === "warming_up") return "Compiling neural pathways...";
    if (status === "loading") {
      if (progress?.status === "progress") {
        const pct = progress.progress?.toFixed(0) || 0;
        const file = progress.file?.split("/").pop() || "";
        return `Loading ${file}... ${pct}%`;
      }
      return LOADING_MESSAGES[Math.floor(Math.random() * LOADING_MESSAGES.length)];
    }
    if (status === "error") return "CONNECTION LOST";
    return "Processing...";
  };

  const pct = (status === "loading" && progress?.status === "progress") ? progress.progress || 0
    : status === "warming_up" ? 100 : null;

  return (
    <div className={`status-bar ${status === "error" ? "status-error" : ""}`}>
      <span className="status-message">{getMessage()}</span>
      {pct !== null && (
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${pct}%` }} />
        </div>
      )}
      {status === "error" && error && (
        <pre className="status-error-detail">{error}</pre>
      )}
    </div>
  );
}
