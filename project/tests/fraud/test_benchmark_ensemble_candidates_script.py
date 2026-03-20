import unittest

import numpy as np
import pandas as pd

from project.scripts.benchmark_ensemble_candidates import (
    build_time_oof_splits,
    feature_hash,
    optimize_weighted_blend,
    rank_candidates,
    resolve_event_time_column,
)


class BenchmarkEnsembleCandidatesScriptTests(unittest.TestCase):
    def test_resolve_event_time_column_defaults_when_missing(self) -> None:
        x = pd.DataFrame({"Amount": [1.0, 2.0]})
        event_time = resolve_event_time_column(x, dataset_source="creditcard")
        self.assertListEqual(event_time.tolist(), [0.0, 0.0])

    def test_build_time_oof_splits_returns_ordered_splits(self) -> None:
        event_time = pd.Series([30, 10, 40, 20, 50], index=[0, 1, 2, 3, 4])
        splits = build_time_oof_splits(event_time, n_splits=3)
        self.assertGreaterEqual(len(splits), 2)
        train_idx, val_idx = splits[0]
        self.assertTrue(np.max(train_idx) != np.max(val_idx))
        self.assertTrue(np.all(np.isin(train_idx, np.array([1, 3]))))

    def test_optimize_weighted_blend_prefers_better_model(self) -> None:
        y = np.array([0, 0, 1, 1])
        scores = {
            "a": np.array([0.1, 0.2, 0.8, 0.9]),
            "b": np.array([0.4, 0.45, 0.55, 0.6]),
        }
        weights, blend = optimize_weighted_blend(scores, y_true=y, threshold=0.5)
        self.assertIn("a", weights)
        self.assertIn("b", weights)
        self.assertEqual(blend.shape[0], y.shape[0])
        self.assertAlmostEqual(weights["a"] + weights["b"], 1.0, places=6)

    def test_feature_hash_stable(self) -> None:
        cols = ["Time", "Amount", "V1"]
        self.assertEqual(feature_hash(cols), feature_hash(cols))

    def test_rank_candidates_prefers_f1_then_pr_auc(self) -> None:
        df = pd.DataFrame(
            [
                {"candidate": "ensemble_equal_weight", "f1": 0.30, "pr_auc": 0.25, "roc_auc": 0.8, "recall": 0.4, "precision": 0.3},
                {"candidate": "ensemble_weighted", "f1": 0.30, "pr_auc": 0.28, "roc_auc": 0.79, "recall": 0.4, "precision": 0.31},
            ]
        )
        ranked = rank_candidates(df, metric="f1")
        self.assertEqual(ranked.iloc[0]["candidate"], "ensemble_weighted")


if __name__ == "__main__":
    unittest.main()
