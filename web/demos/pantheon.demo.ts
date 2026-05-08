import { test, demoType } from '@argo-video/cli';
import { showOverlay, withOverlay, showConfetti } from '@argo-video/cli';
import { trackCursor, cursorHighlight } from '@argo-video/cli';

test('pantheon', async ({ page, narration }) => {
  test.setTimeout(180_000);
  await page.goto('/');
  trackCursor(page, narration);
  cursorHighlight(page, { color: '#4a9a4a', radius: 16 });
  await page.waitForTimeout(1000);

  // ── Scene 1: Hero Page ──
  narration.mark('hero');
  await showOverlay(page, 'hero', narration.durationFor('hero', { maxMs: 7000 }));

  // ── Scene 2: Click Start, Wait for Model ──
  narration.mark('loading');
  await page.click('.hero-cta');
  await page.waitForTimeout(2000);

  // Wait for model to finish loading — overlay shows during the wait
  await withOverlay(page, 'loading', async () => {
    await page.waitForSelector('text=UI ONLINE', { timeout: 120_000 });
    await page.waitForTimeout(narration.durationFor('loading', { minMs: 2000, maxMs: 4000 }));
  });

  // ── Scene 3: First Message ──
  narration.mark('first-message');
  await demoType(page, '.chat-input', 'hey! do you remember the beach?', 50);
  await page.waitForTimeout(500);
  await page.click('.send-button');

  // Wait for response, overlay shows alongside
  await withOverlay(page, 'first-message', async () => {
    await page.waitForTimeout(8000);
    await page.waitForTimeout(narration.durationFor('first-message', { minMs: 3000, maxMs: 6000 }));
  });

  // ── Scene 4: Coffee Message ──
  narration.mark('emotional');
  await demoType(page, '.chat-input', 'would you like to have coffee?', 45);
  await page.waitForTimeout(400);
  await page.click('.send-button');

  await withOverlay(page, 'emotional', async () => {
    await page.waitForTimeout(8000);
    await page.waitForTimeout(narration.durationFor('emotional', { minMs: 3000, maxMs: 6000 }));
  });

  // ── Scene 5: Playful Message ──
  narration.mark('playful');
  await demoType(page, '.chat-input', 'beach or mountains?', 55);
  await page.waitForTimeout(400);
  await page.click('.send-button');

  await page.waitForTimeout(8000);
  await page.waitForTimeout(narration.durationFor('playful', { minMs: 3000, maxMs: 6000 }));

  // ── Scene 6: Toggle the decoder on (round-trip translator) ──
  narration.mark('decode-toggle');
  await page.click('button:has-text("DECODE OFF")');

  // Decoder model loads in a second worker — wait for it to come ready.
  // Post-effect speeds this up in the final cut.
  await withOverlay(page, 'decode-toggle', async () => {
    await page.waitForSelector('button:has-text("DECODE ON")', { timeout: 180_000 });
    await page.waitForTimeout(1500);
    await page.waitForTimeout(narration.durationFor('decode-toggle', { minMs: 3000, maxMs: 6000 }));
  });

  // ── Scene 7: Round-trip — text → emoji → reconstructed text ──
  narration.mark('round-trip');
  await demoType(page, '.chat-input', 'I miss you.', 60);
  await page.waitForTimeout(500);
  await page.click('.send-button');

  await withOverlay(page, 'round-trip', async () => {
    await page.waitForSelector('.reconstruction-block', { timeout: 60_000 });
    // Three samples stream sequentially; give them time to fully populate.
    await page.waitForSelector('.reconstruction-item.status-complete:nth-child(3)', {
      timeout: 120_000,
    }).catch(() => { /* fall through if a sample errors */ });
    await page.waitForTimeout(narration.durationFor('round-trip', { minMs: 6000, maxMs: 10_000 }));
  });

  // ── Scene 8: Closing CTA ──
  narration.mark('closing');
  showConfetti(page, { emoji: ['🧠', '✨', '😊', '🔁'], spread: 'burst', duration: 3000, pieces: 120 });
  await showOverlay(page, 'closing', narration.durationFor('closing', { minMs: 4000, maxMs: 6000 }));
});
