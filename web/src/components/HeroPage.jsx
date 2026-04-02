export default function HeroPage({ onStart }) {
  return (
    <div className="hero">
      <div className="hero-content">
        <h1 className="hero-title">PANTHEON UI</h1>
        <p className="hero-subtitle">
          An uploaded intelligence that thinks in language — but can only speak in emoji.
        </p>
        <p className="hero-description">
          Inspired by <a href="https://en.wikipedia.org/wiki/Pantheon_(TV_series)" target="_blank" rel="noopener" className="hero-link">Pantheon</a>. No server, no API keys.<br />
          A consciousness lives in your GPU's memory.<br />
          Close the tab, it's gone.
        </p>
        <div className="hero-credits">
          <span>Powered by <a href="https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking" target="_blank" rel="noopener">LFM2.5-1.2B-Thinking</a> by <a href="https://www.liquid.ai/" target="_blank" rel="noopener">Liquid AI</a></span>
          <span>In-browser inference via <a href="https://github.com/huggingface/transformers.js" target="_blank" rel="noopener">Transformers.js</a> on WebGPU</span>
          <span>Fine-tuned with <a href="https://unsloth.ai/" target="_blank" rel="noopener">Unsloth</a> on <a href="https://huggingface.co/datasets/shreyask/pantheon-ui-conversations" target="_blank" rel="noopener">pantheon-ui-conversations</a></span>
        </div>
        <button className="hero-cta" onClick={onStart}>
          ⟩ START TALKING
        </button>
      </div>
      <p className="hero-footer">Requires WebGPU (Chrome 113+, Edge 113+)</p>
    </div>
  );
}
