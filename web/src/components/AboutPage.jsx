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
          <h2 className="about-heading">The Round-Trip Translator</h2>
          <p className="about-body">
            Toggle <strong>🔁 DECODE ON</strong> to engage a second model — a decoder fine-tuned
            in the opposite direction. It receives the emoji-only output and reconstructs what was
            originally meant, in plain English. Three samples per emoji string make the lossiness
            of the channel visible: same input, three different decompressions, none of them
            necessarily the original. The drift is the feature.
          </p>
          <p className="about-body">
            Inspired by Anthropic's{" "}
            <a href="https://www.anthropic.com/research/natural-language-autoencoders" target="_blank" rel="noopener">Natural Language Autoencoders</a>{" "}
            — except here the latent space is a sequence of emoji you can actually read.
          </p>
        </section>

        <section className="about-section">
          <h2 className="about-heading">Credits</h2>
          <ul className="about-credits-list">
            <li>Base model: <a href="https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking" target="_blank" rel="noopener">LFM2.5-1.2B-Thinking</a> by <a href="https://www.liquid.ai/" target="_blank" rel="noopener">Liquid AI</a></li>
            <li>Encoder fine-tune: <a href="https://huggingface.co/shreyask/pantheon-ui-onnx" target="_blank" rel="noopener">pantheon-ui-onnx</a> (text → emoji)</li>
            <li>Decoder fine-tune: <a href="https://huggingface.co/shreyask/pantheon-ui-decoder-onnx" target="_blank" rel="noopener">pantheon-ui-decoder-onnx</a> (emoji → text)</li>
            <li>Inference: <a href="https://github.com/huggingface/transformers.js" target="_blank" rel="noopener">Transformers.js v4</a> by <a href="https://huggingface.co/" target="_blank" rel="noopener">Hugging Face</a></li>
            <li>Training: <a href="https://unsloth.ai/" target="_blank" rel="noopener">Unsloth</a> + <a href="https://huggingface.co/docs/trl" target="_blank" rel="noopener">TRL</a> on <a href="https://huggingface.co/docs/huggingface_hub/guides/jobs" target="_blank" rel="noopener">HF Jobs</a></li>
            <li>Datasets: <a href="https://huggingface.co/datasets/shreyask/pantheon-ui-conversations" target="_blank" rel="noopener">pantheon-ui-conversations</a> + <a href="https://huggingface.co/datasets/shreyask/pantheon-ui-decoder-conversations" target="_blank" rel="noopener">pantheon-ui-decoder-conversations</a> (generated with <a href="https://www.anthropic.com/" target="_blank" rel="noopener">Claude API</a>)</li>
            <li>Inspired by <a href="https://en.wikipedia.org/wiki/Pantheon_(TV_series)" target="_blank" rel="noopener">Pantheon</a> (AMC, 2022–2023)</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
