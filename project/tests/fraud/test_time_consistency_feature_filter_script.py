import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project.scripts.time_consistency_feature_filter import (
    build_time_windows,
    load_adversarial_drop_features,
    load_candidate_features,
    load_feature_groups,
    resolve_event_time_column,
)


class TimeConsistencyFeatureFilterScriptTests(unittest.TestCase):
    def test_resolve_event_time_column_defaults_index_sequence(self) -> None:
        frame = pd.DataFrame({"A": [1, 2, 3]})
        series = resolve_event_time_column(frame, dataset_source="creditcard")
        self.assertListEqual(series.tolist(), [0.0, 1.0, 2.0])

    def test_build_time_windows_with_gap(self) -> None:
        event_time = pd.Series(np.arange(20))
        train_idx, val_idx = build_time_windows(event_time, train_fraction=0.6, validation_fraction=0.2)
        self.assertGreater(len(train_idx), 0)
        self.assertGreater(len(val_idx), 0)
        self.assertLess(np.max(train_idx), np.min(val_idx))

    def test_load_feature_groups_filters_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "groups.json"
            path.write_text('{"g1": ["A", "B"], "g2": ["Z"]}', encoding="utf-8")
            groups = load_feature_groups(path, {"A", "B", "C"})
            self.assertIn("g1", groups)
            self.assertNotIn("g2", groups)

    def test_load_candidate_features_respects_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "features.txt"
            path.write_text("A\n#comment\nB\nZ\n", encoding="utf-8")
            selected = load_candidate_features(path, ["A", "B", "C"])
            self.assertListEqual(selected, ["A", "B"])

    def test_load_adversarial_drop_features_filters_by_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report = Path(tmp_dir) / "adversarial_validation_report.json"
            report.write_text(
                '{"ranked_features": [{"feature": "A", "importance_combined": 0.6}, {"feature": "B", "importance_combined": 0.2}]}',
                encoding="utf-8",
            )
            dropped = load_adversarial_drop_features(report, 0.3, {"A", "B", "C"})
            self.assertListEqual(dropped, ["A"])

    def test_results_frame_can_include_label_policy_column(self) -> None:
        frame = pd.DataFrame([{"candidate": "f1", "keep": True}])
        frame["label_policy"] = "transaction"
        self.assertIn("label_policy", frame.columns)
        self.assertEqual(frame.loc[0, "label_policy"], "transaction")


if __name__ == "__main__":
    unittest.main()
