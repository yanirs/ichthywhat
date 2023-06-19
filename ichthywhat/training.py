"""Training and persistence for models that get served."""
from pathlib import Path

import onnx
import torch
import torchvision.transforms
from fastai.learner import load_learner
from fastai.vision.augment import RandomResizedCrop, aug_transforms
from torchvision.models import resnet18

from ichthywhat import experiments
from ichthywhat.constants import DEFAULT_MODELS_PATH


def train_app_model(
    dataset_path: Path, model_version: int, models_path: Path = DEFAULT_MODELS_PATH
) -> None:
    """Train and persist a model with hardcoded settings to be used in the app / api.

    See notebook 03-app.ipynb for usage and output examples.

    Parameters
    ----------
    dataset_path
        training dataset directory.
    model_version
        version of the model. The model will be persisted as app-v{model_version}.pkl
        under models_path.
    models_path
        the directory where the model will be persisted.
    """
    # This import is needed because of some magic patching done by fastai. Training
    # fails without it.
    import fastai.vision.all as _  # noqa: F401

    exported_model_path = models_path / f"app-v{model_version}.pkl"
    if model_version == 1:
        learner = experiments.create_reproducible_learner(
            resnet18,
            dataset_path,
            db_kwargs=dict(splitter=experiments.no_validation_splitter),
            learner_kwargs=dict(cbs=None, metrics=[]),
        )
        experiments.resumable_fine_tune(
            learner, exported_model_path, epochs=120, freeze_epochs=5
        )
    elif model_version == 2:
        learner = experiments.create_reproducible_learner(
            resnet18,
            dataset_path,
            db_kwargs=dict(
                splitter=experiments.no_validation_splitter,
                item_tfms=RandomResizedCrop(448, min_scale=0.5),
                batch_tfms=aug_transforms(mult=2.0),
            ),
            dls_kwargs=dict(bs=16),
            learner_kwargs=dict(cbs=None, metrics=[]),
        )
        experiments.resumable_fine_tune(
            learner, exported_model_path, epochs=200, freeze_epochs=10
        )
    else:
        raise ValueError(f"Unsupported {model_version=}")
    learner.export(exported_model_path)


class _RgbUint8ImgToTensor(torch.nn.Module):
    """Convert uint8 RGB image to a float32 batch in [0, 1].

    Input shape: H x W x C
    Output shape: 1 x C x H x W
    """

    def forward(self, rgb_uint8_img: torch.Tensor) -> torch.Tensor:
        return (rgb_uint8_img.to(torch.float32) / 255).permute(2, 0, 1).unsqueeze(0)


class _SqueezeBatch(torch.nn.Module):
    """Squeeze out the batch to go from shape 1 x classes to a flat array."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.squeeze(0)


def export_learner_to_onnx(learner_path: Path, export_path: Path) -> None:
    """Export learner to an ONNX model and persist it to export_path."""
    learner = load_learner(learner_path)
    # This is a somewhat convoluted way to get the shape of the input, but better than
    # hard-coding it. There may be a better way to inspect the DataLoaders, but it's
    # hard with the usual fastai maze of dynamic attributes and methods.
    batch_size, num_channels, height, width = (
        learner.dls.test_dl([torch.Tensor([0])], num_workers=0).one_batch()[0].shape
    )
    assert batch_size == 1
    assert num_channels == 3
    if height != width:
        raise ValueError(
            "Rectangular inputs require more testing (unsure about the dimension order)"
        )
    model = torch.nn.Sequential(
        # Assume we're working with one RGB image at a time rather than batches.
        _RgbUint8ImgToTensor(),
        # ONNX exports with antialias=True are unsupported with opset_version=16, but
        # antialias=False yields the same performance on the QUT dataset as using PIL's
        # resizing.
        torchvision.transforms.Resize((height, width), antialias=False),
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
        # Remove the batch dimension added by the model to keep the output simple.
        _SqueezeBatch(),
    )
    torch.onnx.export(
        model,
        # Sample input: Height and width don't really matter because they're dynamic.
        torch.randint(
            low=0,
            high=255,
            size=(height // 2, width * 3, num_channels),
            dtype=torch.uint8,
        ),
        str(export_path),
        input_names=["rgb_image"],
        output_names=["class_probabilities"],
        # Allow input images of any size, but still require a fixed number of channels.
        dynamic_axes={"rgb_image": {0: "height", 1: "width"}},
        # Use the maximum supported version.
        opset_version=16,
    )
    # Looks like there isn't an easy way to specify the labels as part of the export.
    # See: https://github.com/pytorch/pytorch/issues/42808
    # Instead, we add a ZipMap that turns the list of predicted probabilities into a
    # mapping from class name to probability.
    onnx_model = onnx.load(str(export_path))
    # Need to explicitly add the opsetid and domain to use ZipMap.
    # See https://stackoverflow.com/a/68504343
    onnx_model.opset_import.append(onnx.helper.make_opsetid("ai.onnx.ml", 1))
    zipmap_node = onnx.helper.make_node(
        "ZipMap",
        inputs=[onnx_model.graph.output[0].name],
        outputs=["class_to_probability"],
        classlabels_strings=list(learner.dls.vocab),
        domain="ai.onnx.ml",
    )
    onnx_model.graph.node.append(zipmap_node)
    # Remove the output info set by torch.onnx.export() and set it to the zipmap_node.
    # See https://github.com/microsoft/onnxruntime/issues/1455#issuecomment-514805365
    onnx_model.graph.output.pop()
    onnx_model.graph.output.append(
        onnx.helper.ValueInfoProto(  # type: ignore[attr-defined]
            name=zipmap_node.output[0]
        )
    )
    onnx.save(onnx_model, str(export_path))
