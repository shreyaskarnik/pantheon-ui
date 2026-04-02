export default function SystemStatus({ status }) {
  const getIndicator = () => {
    switch (status) {
      case "ready": return { color: "var(--status-online)", label: "UI ONLINE" };
      case "generating": return { color: "var(--status-active)", label: "UI TRANSMITTING", pulse: true };
      case "loading": case "warming_up": return { color: "var(--status-loading)", label: "UI BOOTING" };
      case "error": return { color: "var(--status-error)", label: "UI OFFLINE" };
      default: return { color: "var(--status-offline)", label: "UI DORMANT" };
    }
  };
  const { color, label, pulse } = getIndicator();
  return (
    <div className="system-status">
      <span className={`status-dot ${pulse ? "status-pulse" : ""}`} style={{ backgroundColor: color }} />
      <span className="status-label">{label}</span>
    </div>
  );
}
