import type { RuntimeThresholds } from './api.js';

export const DEFAULT_RUNTIME_THRESHOLDS: RuntimeThresholds = {
  approveThreshold: 0.3,
  blockThreshold: 0.7,
};

const formatThreshold = (value: number): string => value.toFixed(2);

export const buildThresholdBandsLabel = (thresholds: RuntimeThresholds): string => {
  const approve = formatThreshold(thresholds.approveThreshold);
  const block = formatThreshold(thresholds.blockThreshold);
  return `APPROVE: score < ${approve} • FLAG: ${approve} ≤ score < ${block} • BLOCK: score ≥ ${block}`;
};
