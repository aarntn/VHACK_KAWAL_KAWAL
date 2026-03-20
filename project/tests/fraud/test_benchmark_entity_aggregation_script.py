import unittest

import numpy as np
import pandas as pd

from project.data.entity_aggregation import (
    EntitySmoothingConfig,
    EntitySmoothingState,
    apply_entity_smoothing_batch,
    smooth_capped_blend,
    smooth_ema,
    smooth_mean,
)


class BenchmarkEntityAggregationScriptTests(unittest.TestCase):
    def test_smooth_mean_groups_by_entity(self) -> None:
        df = pd.DataFrame(
            {
                "entity_id": ["a", "a", "b"],
                "raw_score": [0.2, 0.6, 0.9],
                "event_time": [1, 2, 3],
            }
        )
        mean = smooth_mean(df)
        self.assertAlmostEqual(mean[0], 0.4)
        self.assertAlmostEqual(mean[1], 0.4)
        self.assertAlmostEqual(mean[2], 0.9)

    def test_smooth_ema_respects_entity_boundaries(self) -> None:
        df = pd.DataFrame(
            {
                "entity_id": ["a", "a", "b", "b"],
                "raw_score": [0.0, 1.0, 0.2, 0.8],
                "event_time": [1, 2, 3, 4],
            }
        )
        ema = smooth_ema(df, alpha=0.5)
        self.assertEqual(ema.shape[0], 4)
        self.assertGreater(ema[1], ema[0])
        self.assertGreater(ema[3], ema[2])

    def test_smooth_capped_blend_limits_delta(self) -> None:
        raw = np.array([0.1, 0.5, 0.9])
        agg = np.array([0.9, 0.9, 0.1])
        blended = smooth_capped_blend(raw, agg, alpha=1.0, cap=0.2)
        self.assertTrue(np.all(np.abs(blended - raw) <= 0.200001))

    def test_apply_entity_smoothing_batch_respects_min_history(self) -> None:
        df = pd.DataFrame(
            {
                "entity_id": ["a", "a", "a", "b"],
                "raw_score": [0.2, 0.4, 0.6, 0.9],
            }
        )
        smoothed = apply_entity_smoothing_batch(df, method="mean", min_history=2)
        self.assertAlmostEqual(smoothed[3], 0.9, places=6)  # entity b falls back to raw
        self.assertAlmostEqual(smoothed[0], 0.4, places=6)

    def test_entity_smoothing_state_is_deterministic(self) -> None:
        state = EntitySmoothingState(EntitySmoothingConfig(method="mean", min_history=2))
        s1, d1 = state.smooth("entity-x", 0.8)
        s2, d2 = state.smooth("entity-x", 0.2)
        s3, d3 = state.smooth("entity-x", 0.4)

        self.assertAlmostEqual(s1, 0.8, places=6)
        self.assertAlmostEqual(s2, 0.2, places=6)
        self.assertAlmostEqual(s3, 0.5, places=6)
        self.assertTrue(d1["fallback_used"])
        self.assertTrue(d2["fallback_used"])
        self.assertFalse(d3["fallback_used"])


if __name__ == "__main__":
    unittest.main()
