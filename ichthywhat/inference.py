"""
Thin ONNX wrapper for inference in production.

Originally inspired by https://community.wandb.ai/t/taking-fastai-to-production/1705
"""
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from onnxruntime import InferenceSession
from PIL import Image


class OnnxWrapper:
    """Simple wrapper around an ONNX image classification model."""

    def __init__(self, model_path: Path):
        """Load the ONNX model and prepare for inference."""
        self._ort_sess = InferenceSession(str(model_path))
        self._labels = json.loads(
            self._ort_sess.get_modelmeta().custom_metadata_map["labels"]
        )
        self._input_name = self._ort_sess.get_inputs()[0].name
        self._output_name = self._ort_sess.get_outputs()[0].name

    def predict(self, img: Image.Image) -> pd.Series:
        """Return a series mapping labels to sorted predictions for the image."""
        return pd.Series(
            data=self._ort_sess.run(
                [self._output_name], {self._input_name: np.array(img, dtype=np.uint8)}
            )[0],
            index=self._labels,
        ).sort_values(ascending=False)

    def evaluate(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[str],
        accuracy_top_ks: Sequence[int] = (1, 3, 10),
    ) -> dict[str, float]:
        """Return a mapping from k to accuracy@k for the given paths & labels."""
        # Note: this can be done more efficiently by batching images, but one image at
        # a time is good enough given that this function is only run for evaluation
        # purposes. Also, batch support was removed from the model for simplicity.
        correct_at_k = {k: 0 for k in accuracy_top_ks}
        for image_path, label in zip(image_paths, labels, strict=True):
            predictions = self.predict(Image.open(image_path))
            for k in accuracy_top_ks:
                if label in predictions[:k].index:
                    correct_at_k[k] += 1
        return {
            f"top_{k}_accuracy": correct / len(image_paths)
            for k, correct in correct_at_k.items()
        }
