"""
ONNX export and inference proof-of-concept.

Builds on https://community.wandb.ai/t/taking-fastai-to-production/1705
"""
import io
import json
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import torch
import torchvision.transforms
from fastai.learner import Learner
from onnxruntime import InferenceSession
from PIL import Image

# TODO: validate exported onnx model on the QUT dataset
# TODO: refactor streamlit app (use functions) and change it to use ONNX
# TODO: use ONNX in the API (rewrite Dockerfile accordingly)
# TODO: try running inference in the browser


# TODO: move this to models.py to make OnnxWrapper independent of fastai & torch.
def export_learner_to_onnx(learner: Learner, export_path: Path) -> None:
    """Export learner to an ONNX model and persist it to export_path."""
    model = torch.nn.Sequential(
        # TODO: add more steps from OnnxWrapper._load_input_image()?
        # Replace fastai's normalisation. Tested only with resnet18 (these are ImageNet
        # stats), so other models might not have the step.
        torchvision.transforms.Normalize(
            mean=learner.dls.after_batch.normalize.mean.squeeze(),
            std=learner.dls.after_batch.normalize.std.squeeze(),
        ),
        # Actual PyTorch model.
        learner.model.eval(),
        # Replace fastai's softmax layer.
        torch.nn.Softmax(dim=1),
    )
    # This is a somewhat convoluted way to get the shape of the input, but better than
    # hard-coding it. There may be a better way to inspect the DataLoaders, but it's
    # hard with the usual fastai maze of dynamic attributes and methods.
    input_shape = (
        learner.dls.test_dl([torch.Tensor([0])], num_workers=0).one_batch()[0].shape
    )
    if input_shape[-1] != input_shape[-2]:
        raise ValueError(
            "Rectangular images require more tests (see OnnxWrapper._load_input_image)"
        )
    torch.onnx.export(
        model,
        torch.randn(input_shape),
        str(export_path),
        input_names=["input"],
        output_names=["output"],
        # Allow variable length batches, but still require fixed image sizes.
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        # Use the maximum supported version.
        opset_version=16,
    )
    # Looks like there isn't an easy way to specify the labels as part of the export.
    # See: https://github.com/pytorch/pytorch/issues/42808
    onnx_model = onnx.load(str(export_path))
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="labels", value=json.dumps(list(learner.dls.vocab))
        )
    )
    onnx.save(onnx_model, str(export_path))


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

    def _load_input_image(self, path_or_file: Path | io.BytesIO) -> np.ndarray:
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
