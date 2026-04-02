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
        <button className="hero-cta" onClick={onStart}>
          ⟩ START TALKING
        </button>
      </div>
      <p className="hero-footer">Requires WebGPU (Chrome 113+, Edge 113+)</p>
    </div>
  );
}
