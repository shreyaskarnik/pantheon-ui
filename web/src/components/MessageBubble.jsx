import ThinkingTrace from "./ThinkingTrace.jsx";
import EmojiResponse from "./EmojiResponse.jsx";
import Reconstruction from "./Reconstruction.jsx";

export default function MessageBubble({ message, showThinking, isLatest }) {
  if (message.role === "user") {
    return <div className="message message-user"><p>{message.content}</p></div>;
  }
  if (message.role === "assistant") {
    return (
      <div className="message message-assistant">
        <ThinkingTrace text={message.thinking} isComplete={!!message.emoji} showThinking={showThinking} />
        <EmojiResponse emoji={message.emoji} animate={isLatest} />
        <Reconstruction
          originalText={message.originalText}
          reconstructions={message.reconstructions}
          showThinking={showThinking}
        />
        {!message.emoji && !message.thinking && <div className="generating-indicator">🧠</div>}
      </div>
    );
  }
  return null;
}
