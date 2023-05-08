"""Training and persistence for models that get served."""

from pathlib import Path

from fastai.vision.augment import RandomResizedCrop, aug_transforms
from torchvision.models import resnet18

from ichthywhat import experiments


def train_app_model(dataset_path: Path, models_path: Path, model_version: int) -> None:
    """
    Train and persist a model with hardcoded settings to be used in the app / api.

    See notebook 03-app.ipynb for usage and output examples.

    :param dataset_path: path to the dataset.
    :param models_path: path to the directory to persist the model.
    :param model_version: version of the model. The model will be persisted as
                          app-v{model_version}.pkl under models_path.
    """
    if model_version == 1:
        learner = experiments.create_reproducible_learner(
            resnet18,
            dataset_path,
            db_kwargs=dict(splitter=experiments.no_validation_splitter),
            learner_kwargs=dict(cbs=None),
        )
        learner.fine_tune(120, freeze_epochs=5)
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
            learner_kwargs=dict(cbs=None),
        )
        learner.fine_tune(200, freeze_epochs=10)
    else:
        raise ValueError(f"Unsupported {model_version=}")
    learner.export(models_path / f"app-v{model_version}.pkl")