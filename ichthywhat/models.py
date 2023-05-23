"""Training and persistence for models that get served."""

from pathlib import Path

from fastai.vision.augment import RandomResizedCrop, aug_transforms
from torchvision.models import resnet18

from ichthywhat import experiments
from ichthywhat.constants import DEFAULT_MODELS_PATH


def train_app_model(
    dataset_path: Path, model_version: int, models_path: Path = DEFAULT_MODELS_PATH
) -> None:
    """
    Train and persist a model with hardcoded settings to be used in the app / api.

    See notebook 03-app.ipynb for usage and output examples.

    :param dataset_path: path to the dataset.
    :param model_version: version of the model. The model will be persisted as
                          app-v{model_version}.pkl under models_path.
    :param models_path: path to the directory to persist the model.
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
