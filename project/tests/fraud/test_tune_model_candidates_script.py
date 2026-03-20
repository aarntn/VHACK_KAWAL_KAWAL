import unittest

import pandas as pd

from project.scripts.tune_model_candidates import (
    append_ensemble_to_robustness,
    groupkfold_uid_split,
    encode_categorical_columns,
    assign_time_groups,
    build_grouped_folds,
    build_temporal_windows,
    parse_window_configs,
    rank_candidates,
    select_overall_best,
    resolve_event_time_column,
    time_based_split,
)


class TuneModelCandidatesScriptTests(unittest.TestCase):
    def test_time_based_split_orders_by_event_time(self) -> None:
        x = pd.DataFrame({"a": [1, 2, 3, 4]}, index=[10, 11, 12, 13])
        y = pd.Series([0, 1, 0, 1], index=x.index)
        event_time = pd.Series([300, 100, 400, 200], index=x.index)

        x_train, x_test, y_train, y_test = time_based_split(x, y, event_time, test_size=0.5)

        self.assertEqual(list(x_train.index), [11, 13])
        self.assertEqual(list(x_test.index), [10, 12])
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_test), 2)

    def test_resolve_event_time_column_defaults_when_missing(self) -> None:
        x = pd.DataFrame({"Amount": [1.0, 2.0]})
        event_time = resolve_event_time_column(x, dataset_source="creditcard")
        self.assertListEqual(event_time.tolist(), [0.0, 0.0])

    def test_rank_candidates_prefers_primary_metric_then_pr_auc(self) -> None:
        df = pd.DataFrame(
            [
                {"candidate": "a", "f1": 0.31, "pr_auc": 0.21, "roc_auc": 0.9, "recall": 0.4, "precision": 0.3},
                {"candidate": "b", "f1": 0.31, "pr_auc": 0.25, "roc_auc": 0.88, "recall": 0.39, "precision": 0.31},
                {"candidate": "c", "f1": 0.30, "pr_auc": 0.30, "roc_auc": 0.92, "recall": 0.42, "precision": 0.28},
            ]
        )

        ranked = rank_candidates(df, metric="f1")
        self.assertEqual(ranked.iloc[0]["candidate"], "b")


    def test_parse_window_configs(self) -> None:
        configs = parse_window_configs("4:1:1,2:0:1")
        self.assertEqual(configs, [(4, 1, 1), (2, 0, 1)])

    def test_build_temporal_and_grouped_windows(self) -> None:
        event_time = pd.Series(range(30))
        group_ids = assign_time_groups(event_time, n_groups=6)
        holdouts = build_temporal_windows(group_ids, configs=[(2, 1, 1)])
        folds = build_grouped_folds(group_ids, gap_groups=1)
        self.assertGreater(len(holdouts), 0)
        self.assertGreater(len(folds), 0)

    def test_select_overall_best_prefers_ensemble_when_metric_higher(self) -> None:
        single = {"candidate": "xgboost_tuned", "f1": 0.30, "pr_auc": 0.20, "roc_auc": 0.80, "recall": 0.3, "precision": 0.3}
        ensemble = {"candidate": "ensemble_weighted", "f1": 0.35, "pr_auc": 0.22, "roc_auc": 0.81, "recall": 0.31, "precision": 0.34, "_source_type": "ensemble_benchmark"}
        overall = select_overall_best(single, ensemble)
        self.assertEqual(overall["candidate"], "ensemble_weighted")
        self.assertEqual(overall["candidate_origin"], "ensemble")

    def test_append_ensemble_to_robustness_schema_stable(self) -> None:
        base = {"xgboost_tuned": {"window_count": 2, "validation_modes": ["temporal_holdout"], "metrics": {"f1": {"mean": 0.3, "std": 0.0, "worst": 0.3}}}}
        ensemble = {"candidate": "ensemble_weighted", "f1": 0.4, "pr_auc": 0.2, "roc_auc": 0.8, "recall": 0.5, "precision": 0.6, "false_positive_rate": 0.1}
        merged = append_ensemble_to_robustness(base, ensemble)
        key = "ensemble::ensemble_weighted"
        self.assertIn(key, merged)
        self.assertIn("metrics", merged[key])
        self.assertIn("f1", merged[key]["metrics"])
        self.assertEqual(merged[key]["source"], "ensemble_benchmark")

    def test_groupkfold_uid_split_returns_non_empty_partitions(self) -> None:
        x = pd.DataFrame(
            {
                "TransactionDT": [0, 1, 2, 3, 4, 5],
                "card1": [1, 1, 2, 2, 3, 3],
                "addr1": [10, 10, 20, 20, 30, 30],
                "f": [0, 1, 0, 1, 0, 1],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1])
        x_train, x_test, y_train, y_test = groupkfold_uid_split(x, y, dataset_source="ieee_cis", n_splits=3)
        self.assertGreater(len(x_train), 0)
        self.assertGreater(len(x_test), 0)
        self.assertGreaterEqual(y_train.nunique(), 2)
        self.assertGreaterEqual(y_test.nunique(), 2)

    def test_encode_categorical_columns_converts_object_to_codes(self) -> None:
        x = pd.DataFrame({"a": ["x", "y", None], "b": [1.0, 2.0, 3.0]})
        out = encode_categorical_columns(x)
        self.assertTrue(pd.api.types.is_integer_dtype(out["a"]))
        self.assertTrue(pd.api.types.is_float_dtype(out["b"]))


if __name__ == "__main__":
    unittest.main()
