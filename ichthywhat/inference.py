"""
Thin ONNX wrapper for inference in production.

Originally inspired by https://community.wandb.ai/t/taking-fastai-to-production/1705
"""
import io
import json
from pathlib import Path
from typing import Any

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
        self._img_size = self._ort_sess.get_inputs()[0].shape[2:]
        self._input_name = self._ort_sess.get_inputs()[0].name
        self._output_name = self._ort_sess.get_outputs()[0].name

    def predict(self, img_path_or_file: Path | io.BytesIO) -> pd.Series:
        """Return a series mapping labels to sorted predictions for the image."""
        img = self._load_input_image(img_path_or_file)
        return pd.Series(
            data=self._ort_sess.run([self._output_name], {self._input_name: img})[0][0],
            index=self._labels,
        ).sort_values(ascending=False)

    def _load_input_image(
        self, path_or_file: Path | io.BytesIO
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        # Starting from an RGB image with uint8 values, resize it to self._img_size and
        # transform it to end up with an array of shape (1, 3, width, height) with
        # float32 values in the [0, 1] range.
        # TODO: the transpose step may be wrong for rectangular images (see
        # TODO: torchvision.transforms.Resize versus PIL's resize)
        return (
            (
                np.array(
                    Image.open(path_or_file).resize(self._img_size), dtype=np.float32
                )
                / 255
            )
            .transpose(2, 0, 1)
            .reshape((1, 3, *self._img_size))
        )
