"""Training and persistence for models that get served."""
import json
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


def export_learner_to_onnx(learner_path: Path, export_path: Path) -> None:
    """Export learner to an ONNX model and persist it to export_path."""
    learner = load_learner(learner_path)
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
