import assert from 'node:assert/strict';
import test from 'node:test';
import { buildThresholdBandsLabel } from '../src/thresholds.js';

test('threshold labels change when backend threshold values change', async () => {
  const firstMockedBackendThresholds = { approveThreshold: 0.2, blockThreshold: 0.8 };
  const secondMockedBackendThresholds = { approveThreshold: 0.35, blockThreshold: 0.65 };

  const firstLabel = buildThresholdBandsLabel(firstMockedBackendThresholds);
  const secondLabel = buildThresholdBandsLabel(secondMockedBackendThresholds);

  assert.match(firstLabel, /APPROVE: score < 0.20/);
  assert.match(firstLabel, /BLOCK: score ≥ 0.80/);
  assert.match(secondLabel, /APPROVE: score < 0.35/);
  assert.match(secondLabel, /BLOCK: score ≥ 0.65/);
});
