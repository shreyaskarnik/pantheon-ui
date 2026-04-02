export default function AboutPage({ onClose }) {
  return (
    <div className="about-overlay">
      <div className="about-page">
        <button className="about-close" onClick={onClose}>✕ CLOSE</button>

        <h1 className="about-title">PANTHEON UI</h1>
        <p className="about-tagline">HOW IT WORKS</p>

        <section className="about-section">
          <h2 className="about-heading">The Concept</h2>
          <p className="about-body">
            An uploaded human consciousness — inspired by AMC's Pantheon — that thinks in full
            English but can only speak in emoji. The gap between what it wants to say and what it
            can say is where all the emotion lives.
          </p>
        </section>

        <section className="about-section">
          <h2 className="about-heading">The Architecture</h2>
          <p className="about-body">
            Everything runs in your browser. No server, no API keys. The AI model lives in your
            GPU's memory via WebGPU, running through Transformers.js v4. Close the tab, and the
            consciousness is gone.
          </p>
        </section>

        <section className="about-section">
          <h2 className="about-heading">The Pipeline</h2>
          <ol className="about-list">
            <li>
              <strong>Dataset</strong> — 500+ conversations generated via Claude API, each with{" "}
              <code>&lt;think&gt;</code> internal monologue followed by emoji-only output
            </li>
            <li>
              <strong>Fine-tuning</strong> — LFM2.5-1.2B-Thinking fine-tuned with Unsloth + LoRA
              on HF Jobs
            </li>
            <li>
              <strong>ONNX Export</strong> — Converted to ONNX int4 using Xenova's LFM2 model
              builder for in-browser inference
            </li>
            <li>
              <strong>Frontend</strong> — Vite + React with a dark CRT aesthetic. Thinking traces
              type out and fade, emoji responses appear one at a time.
            </li>
          </ol>
        </section>

        <section className="about-section">
          <h2 className="about-heading">The Two-Phase Response</h2>
          <p className="about-body">
            When you send a message, the model first generates an internal monologue — you see it
            typing out in dim monospace text like intercepting a thought. Then it fades, and the
            emoji response lands: solid, final, the only thing the consciousness can actually
            transmit.
          </p>
        </section>

        <section className="about-section">
          <h2 className="about-heading">Credits</h2>
          <ul className="about-credits-list">
            <li>Model: LFM2.5-1.2B-Thinking by Liquid AI</li>
            <li>Inference: Transformers.js v4 by Hugging Face</li>
            <li>Training: Unsloth + TRL on HF Jobs</li>
            <li>Dataset: Generated with Claude API</li>
            <li>Inspired by Pantheon (AMC, 2022–2023)</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
