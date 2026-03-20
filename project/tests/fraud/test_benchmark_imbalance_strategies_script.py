import unittest

import numpy as np
import pandas as pd

from project.scripts.benchmark_imbalance_strategies import (
    add_stability_columns,
    apply_resampling,
    build_expanding_time_windows,
    encode_categorical_columns,
    evaluate_strategy,
    false_positive_rate,
    prepare_resampling_features,
    summarize_stability,
    time_based_split,
)


class BenchmarkImbalanceStrategiesScriptTests(unittest.TestCase):
    def test_time_based_split_orders_by_event_time(self) -> None:
        x = pd.DataFrame({"a": [1, 2, 3, 4]}, index=[10, 11, 12, 13])
        y = pd.Series([0, 1, 0, 1], index=x.index)
        event_time = pd.Series([300, 100, 400, 200], index=x.index)

        x_train, x_test, y_train, y_test = time_based_split(x, y, event_time, test_size=0.5)

        self.assertEqual(list(x_train.index), [11, 13])
        self.assertEqual(list(x_test.index), [10, 12])
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_test), 2)

    def test_false_positive_rate(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0])
        self.assertAlmostEqual(false_positive_rate(y_true, y_pred), 1 / 3, places=6)

    def test_apply_resampling_baseline_no_change(self) -> None:
        x_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, 0.0, 1.0]})
        y_train = pd.Series([0, 0, 1])

        x_out, y_out = apply_resampling("baseline", x_train, y_train, seed=42)

        self.assertEqual(x_out.shape, x_train.shape)
        self.assertEqual(y_out.shape, y_train.shape)
        self.assertListEqual(x_out.columns.tolist(), x_train.columns.tolist())

    def test_build_expanding_time_windows_returns_ordered_unique_windows(self) -> None:
        event_time = pd.Series([50, 10, 30, 20, 40], index=[0, 1, 2, 3, 4])
        windows = build_expanding_time_windows(event_time, test_size=0.4, n_windows=3)

        self.assertGreaterEqual(len(windows), 2)
        prior_last_train = -1
        for train_order, test_order in windows:
            self.assertGreater(len(train_order), 0)
            self.assertGreater(len(test_order), 0)
            self.assertTrue(set(train_order).isdisjoint(set(test_order)))
            self.assertLessEqual(event_time.iloc[train_order].max(), event_time.iloc[test_order].min())
            self.assertGreaterEqual(len(train_order), prior_last_train + 1)
            prior_last_train = len(train_order)

    def test_summarize_stability_calculates_mean_std_worst(self) -> None:
        window_results = [
            type("R", (), {"f1": 0.2, "pr_auc": 0.3, "recall": 0.4})(),
            type("R", (), {"f1": 0.4, "pr_auc": 0.5, "recall": 0.6})(),
            type("R", (), {"f1": 0.6, "pr_auc": 0.7, "recall": 0.8})(),
        ]

        summary = summarize_stability(window_results)

        self.assertEqual(summary["stability_window_count"], 3)
        self.assertAlmostEqual(summary["f1_mean"], 0.4, places=6)
        self.assertAlmostEqual(summary["pr_auc_mean"], 0.5, places=6)
        self.assertAlmostEqual(summary["recall_mean"], 0.6, places=6)
        self.assertAlmostEqual(summary["f1_worst_window"], 0.2, places=6)

    def test_add_stability_columns_adds_ranks_deterministically(self) -> None:
        df = pd.DataFrame(
            {
                "strategy": ["a", "b", "c"],
                "f1_mean": [0.5, 0.4, 0.3],
                "f1_std": [0.05, 0.04, 0.03],
                "f1_worst_window": [0.45, 0.35, 0.25],
                "pr_auc_mean": [0.6, 0.5, 0.4],
                "pr_auc_std": [0.03, 0.04, 0.05],
                "pr_auc_worst_window": [0.55, 0.45, 0.35],
                "recall_mean": [0.7, 0.6, 0.5],
                "recall_std": [0.05, 0.05, 0.05],
                "recall_worst_window": [0.65, 0.55, 0.45],
            }
        )

        out = add_stability_columns(df)

        self.assertIn("rank_pr_auc_mean", out.columns)
        self.assertIn("rank_recall_worst", out.columns)
        self.assertIn("robustness_score", out.columns)
        self.assertIn("rank_robustness_score", out.columns)
        ranks = out.set_index("strategy")["rank_robustness_score"].to_dict()
        self.assertEqual(ranks["a"], 1)
        self.assertGreater(ranks["c"], ranks["a"])

    def test_apply_resampling_single_class_returns_baseline(self) -> None:
        x_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0]})
        y_train = pd.Series([0, 0, 0])

        x_out, y_out = apply_resampling("smote", x_train, y_train, seed=42)

        self.assertTrue(x_out.equals(x_train))
        self.assertTrue(y_out.equals(y_train))

    def test_evaluate_strategy_single_class_train_does_not_raise(self) -> None:
        x_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, 0.0, 1.0]})
        y_train = pd.Series([0, 0, 0])
        x_test = pd.DataFrame({"f1": [0.5, 1.5], "f2": [1.0, 0.0]})
        y_test = pd.Series([0, 1])

        result = evaluate_strategy(
            strategy="smote",
            dataset_source="creditcard",
            split_mode="time",
            threshold=0.5,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=42,
        )

        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 1)
        self.assertEqual(result.tn, 1)

    def test_encode_categorical_columns_converts_object_to_codes(self) -> None:
        x = pd.DataFrame({"cat": ["a", "b", None], "num": [1.0, 2.0, 3.0]})
        out = encode_categorical_columns(x)
        self.assertTrue(pd.api.types.is_integer_dtype(out["cat"]))
        self.assertTrue(pd.api.types.is_float_dtype(out["num"]))

    def test_prepare_resampling_features_replaces_nan_and_inf(self) -> None:
        x = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "b": [np.inf, 2.0, -np.inf],
            }
        )
        out = prepare_resampling_features(x)
        self.assertFalse(np.isnan(out.to_numpy(dtype=float)).any())
        self.assertTrue(np.isfinite(out.to_numpy(dtype=float)).all())


if __name__ == "__main__":
    unittest.main()
