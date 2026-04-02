import { defineConfig } from "@argo-video/cli";

export default defineConfig({
  baseURL: "http://localhost:5173",
  demosDir: "demos",
  outputDir: "videos",
  tts: { defaultVoice: "af_heart", defaultSpeed: 0.92 },
  video: {
    width: 1920,
    height: 1080,
    fps: 30,
    browser: "chromium",
    deviceScaleFactor: 1,
    launchOptions: {
      channel: "chrome",
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,WebGPU",
        "--use-angle=metal",
      ],
    },
  },
  export: {
    preset: "slow",
    crf: 18,
    transition: { type: "fade-through-black", durationMs: 1500 },
    audio: { loudnorm: true },
    sharpen: true,
  },
  overlays: {
    autoBackground: true,
  },
});
