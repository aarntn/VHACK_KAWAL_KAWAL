import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project.scripts.adversarial_validation import (
    build_adversarial_dataset,
    compute_ranked_importance,
    derive_drop_features,
    resolve_event_time_column,
    time_based_split_indices,
)


class AdversarialValidationScriptTests(unittest.TestCase):
    def test_time_based_split_indices(self) -> None:
        event_time = pd.Series([30, 10, 20, 40])
        train_idx, test_idx = time_based_split_indices(event_time, test_size=0.5)
        self.assertEqual(train_idx.tolist(), [1, 2])
        self.assertEqual(test_idx.tolist(), [0, 3])

    def test_resolve_event_time_column_fallback(self) -> None:
        frame = pd.DataFrame({"A": [1, 2, 3]})
        event_time = resolve_event_time_column(frame, dataset_source="creditcard")
        self.assertListEqual(event_time.tolist(), [0.0, 1.0, 2.0])

    def test_build_dataset_from_explicit_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train = Path(tmp) / "train.csv"
            test = Path(tmp) / "test.csv"
            pd.DataFrame({"f1": [0, 0], "f2": [1, 1], "Class": [0, 1]}).to_csv(train, index=False)
            pd.DataFrame({"f1": [1, 1], "f2": [0, 0], "Class": [1, 0]}).to_csv(test, index=False)

            class Args:
                train_csv = train
                test_csv = test
                dataset_source = "creditcard"
                label_policy = "transaction"
                split_mode = "time"
                test_size = 0.2
                seed = 42
                dataset_path = Path("x")
                ieee_transaction_path = None
                ieee_identity_path = None

            x_adv, y_adv, meta = build_adversarial_dataset(Args())

        self.assertEqual(x_adv.shape, (4, 2))
        self.assertListEqual(y_adv.tolist(), [0, 0, 1, 1])
        self.assertEqual(meta["split_source"], "explicit_files")
        self.assertEqual(meta["label_policy"], "transaction")

    def test_compute_ranked_importance_deterministic_top_feature(self) -> None:
        rng = np.random.default_rng(42)
        drift_feature = np.concatenate([np.zeros(120), np.ones(120)])
        noise = rng.normal(size=240)
        x_adv = pd.DataFrame({"drift": drift_feature, "noise": noise})
        y_adv = np.concatenate([np.zeros(120, dtype=int), np.ones(120, dtype=int)])

        ranked, metrics = compute_ranked_importance(x_adv, y_adv, seed=42)

        self.assertEqual(ranked.iloc[0]["feature"], "drift")
        self.assertGreater(float(metrics["logistic"]["roc_auc"]), 0.9)
        self.assertGreater(float(metrics["tree"]["roc_auc"]), 0.9)

    def test_derive_drop_features(self) -> None:
        ranked = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "importance_combined": [0.5, 0.2, 0.1],
            }
        )
        self.assertEqual(derive_drop_features(ranked, None), [])
        self.assertEqual(derive_drop_features(ranked, 0.2), ["a", "b"])

    def test_ranked_importance_can_attach_label_policy_column(self) -> None:
        x_adv = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1, 2, 3, 4]})
        y_adv = np.array([0, 0, 1, 1])
        ranked, _ = compute_ranked_importance(x_adv, y_adv, seed=42)
        ranked["label_policy"] = "transaction"
        self.assertIn("label_policy", ranked.columns)
        self.assertEqual(ranked["label_policy"].iloc[0], "transaction")


if __name__ == "__main__":
    unittest.main()
