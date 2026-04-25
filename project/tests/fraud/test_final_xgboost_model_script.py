import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

try:
    from project.models import final_xgboost_model
except ModuleNotFoundError:
    import models.final_xgboost_model as final_xgboost_model


class FinalXGBoostModelScriptTests(unittest.TestCase):
    def _base_args(self, **overrides):
        args = {
            "model_output": None,
            "features_output": None,
            "thresholds_output": None,
            "preprocessing_artifact_output": None,
            "use_preprocessing": False,
            "overwrite_runtime_artifacts": False,
        }
        args.update(overrides)
        return Namespace(**args)

    def test_preprocessing_defaults_route_to_preprocessed_outputs(self):
        args = self._base_args(use_preprocessing=True)

        model_output, features_output, thresholds_output, preprocessing_output = (
            final_xgboost_model._resolve_output_targets(args)
        )

        self.assertEqual(model_output, final_xgboost_model.DEFAULT_PREPROCESSED_MODEL_PATH)
        self.assertEqual(features_output, final_xgboost_model.DEFAULT_PREPROCESSED_FEATURES_PATH)
        self.assertEqual(thresholds_output, final_xgboost_model.DEFAULT_PREPROCESSED_THRESHOLDS_PATH)
        self.assertEqual(preprocessing_output, final_xgboost_model.DEFAULT_PREPROCESSING_ARTIFACT_PATH)

    def test_preprocessing_respects_explicit_runtime_paths(self):
        args = self._base_args(
            use_preprocessing=True,
            model_output=str(final_xgboost_model.DEFAULT_MODEL_PATH),
            features_output=str(final_xgboost_model.DEFAULT_FEATURES_PATH),
            thresholds_output=str(final_xgboost_model.DEFAULT_THRESHOLDS_PATH),
            preprocessing_artifact_output=str(final_xgboost_model.DEFAULT_PREPROCESSING_ARTIFACT_PATH),
        )

        model_output, features_output, thresholds_output, preprocessing_output = (
            final_xgboost_model._resolve_output_targets(args)
        )

        self.assertEqual(model_output, final_xgboost_model.DEFAULT_MODEL_PATH)
        self.assertEqual(features_output, final_xgboost_model.DEFAULT_FEATURES_PATH)
        self.assertEqual(thresholds_output, final_xgboost_model.DEFAULT_THRESHOLDS_PATH)
        self.assertEqual(preprocessing_output, final_xgboost_model.DEFAULT_PREPROCESSING_ARTIFACT_PATH)

    def test_preprocessing_allows_runtime_overwrite_flag(self):
        args = self._base_args(use_preprocessing=True, overwrite_runtime_artifacts=True)

        model_output, features_output, thresholds_output, _ = final_xgboost_model._resolve_output_targets(args)

        self.assertEqual(model_output, final_xgboost_model.DEFAULT_MODEL_PATH)
        self.assertEqual(features_output, final_xgboost_model.DEFAULT_FEATURES_PATH)
        self.assertEqual(thresholds_output, final_xgboost_model.DEFAULT_THRESHOLDS_PATH)

    def test_load_feature_whitelist_from_json_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "decisions.json"
            path.write_text(json.dumps({"kept_single_features": ["V1", "V2"]}), encoding="utf-8")
            loaded = final_xgboost_model._load_feature_whitelist(str(path))
            self.assertEqual(loaded, ["V1", "V2"])

    def test_load_feature_whitelist_from_time_consistency_decisions_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "decisions.json"
            path.write_text(
                json.dumps({"summary": {"kept_single_features": ["V3", "V4"]}}),
                encoding="utf-8",
            )
            loaded = final_xgboost_model._load_feature_whitelist(str(path))
            self.assertEqual(loaded, ["V3", "V4"])

    def test_load_feature_whitelist_from_feature_whitelist_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "whitelist.json"
            path.write_text(json.dumps({"feature_whitelist": ["V5", "V6"]}), encoding="utf-8")
            loaded = final_xgboost_model._load_feature_whitelist(str(path))
            self.assertEqual(loaded, ["V5", "V6"])

    def test_load_feature_whitelist_raises_for_unsupported_json_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "unsupported.json"
            path.write_text(json.dumps({"summary": {"kept_single_features": []}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Accepted JSON keys"):
                final_xgboost_model._load_feature_whitelist(str(path))

    def test_load_feature_whitelist_allows_empty_when_flag_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "unsupported.json"
            path.write_text(json.dumps({"summary": {"kept_single_features": []}}), encoding="utf-8")
            loaded = final_xgboost_model._load_feature_whitelist(str(path), allow_empty=True)
            self.assertIsNone(loaded)

    def test_apply_feature_whitelist_raises_on_missing(self):
        import pandas as pd

        frame = pd.DataFrame({"A": [1], "B": [2]})
        with self.assertRaises(ValueError):
            final_xgboost_model._apply_feature_whitelist(frame, ["A", "C"])

    def test_apply_feature_whitelist_filters_columns(self):
        import pandas as pd

        frame = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        filtered = final_xgboost_model._apply_feature_whitelist(frame, ["C", "A"])
        self.assertListEqual(filtered.columns.tolist(), ["C", "A"])

    def test_load_adversarial_drop_features_from_ranked_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "adv_report.json"
            path.write_text(
                json.dumps(
                    {
                        "ranked_features": [
                            {"feature": "V1", "importance_combined": 0.7},
                            {"feature": "V2", "importance_combined": 0.1},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loaded = final_xgboost_model._load_adversarial_drop_features(str(path), max_importance=0.2)
            self.assertEqual(loaded, ["V1"])

    def test_drop_adversarial_features(self):
        import pandas as pd

        frame = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
        filtered = final_xgboost_model._drop_adversarial_features(frame, ["B"])
        self.assertListEqual(filtered.columns.tolist(), ["A", "C"])


if __name__ == "__main__":
    unittest.main()
