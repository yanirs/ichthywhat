"""Assorted experiment utilities."""
from pathlib import Path

import mlflow
from fastai.callback.core import Callback
from fastai.callback.mixup import MixUp
from fastai.data.block import CategoryBlock, DataBlock
from fastai.data.transforms import RandomSplitter, get_image_files
from fastai.learner import Learner
from fastai.metrics import top_k_accuracy, accuracy
from fastai.torch_core import set_seed
from fastai.vision.augment import aug_transforms, RandomResizedCrop
from fastai.vision.data import ImageBlock
from fastai.vision.learner import cnn_learner


class MLflowCallback(Callback):
    """A Learner callback that logs the metrics of each epoch to MLflow."""

    def __init__(self, run_name_prefix="", **kwargs):
        super().__init__(**kwargs)
        self.run_name_prefix = run_name_prefix

    def before_epoch(self):
        # If the recorder doesn't have a log yet, it's the initial run (before epoch 0).
        epoch = self.epoch if hasattr(self.recorder, "log") else 0
        mlflow.start_run(
            run_name=f'{self.run_name_prefix}{"frozen" if self.opt.frozen_idx else "unfrozen"} epoch {epoch}',
            nested=True,
        )

    def after_epoch(self):
        mlflow.log_metrics(dict(zip(self.recorder.metric_names[1:], self.recorder.log[1:])))
        mlflow.end_run()


def top_3_accuracy(inp, targ):
    """Delegate to top_k_accuarcy with k=3."""
    return top_k_accuracy(inp, targ, k=3)


def top_10_accuracy(inp, targ):
    """Delegate to top_k_accuarcy with k=10."""
    return top_k_accuracy(inp, targ, k=10)


def delete_run_with_children(parent_run_id: str):
    """Delete an MLflow run together with all its children."""
    client = mlflow.tracking.client.MlflowClient()
    parent_run = client.get_run(parent_run_id)
    for child_run in client.search_runs(
        experiment_ids=[parent_run.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'"
    ):
        client.delete_run(child_run.info.run_id)
    client.delete_run(parent_run_id)


def get_species_from_path(path: Path) -> str:
    """Get the species name from a path, assuming that the file is named `<genus>-<taxon>-.*`."""
    return " ".join(path.name.split("-")[:2]).capitalize()


def create_reproducible_learner(arch, dataset_path: Path, db_kwargs=None, dls_kwargs=None, learner_kwargs=None):
    """
    Create a learner that should yield reproducible results across runs.

    :param arch: the architecture
    :param dataset_path: the path of the dataset
    :param db_kwargs: optional keyword arguments for the DataBlock
    :param dls_kwargs: optional keyword arguments for the DataLoaders
    :param learner_kwargs: optional keyword arguments for the Learner
    :return: the learner, as produced by `cnn_learner()`
    """
    # See https://github.com/fastai/fastai/issues/2832#issuecomment-698759541
    set_seed(42, reproducible=True)
    dls = DataBlock(
        **{
            **dict(
                blocks=(ImageBlock, CategoryBlock),
                get_items=get_image_files,
                splitter=RandomSplitter(valid_pct=0.2, seed=42),
                get_y=get_species_from_path,
                item_tfms=RandomResizedCrop(224, min_scale=0.5),
                batch_tfms=aug_transforms(),
            ),
            **(db_kwargs or {}),
        }
    ).dataloaders(dataset_path, **(dls_kwargs or {}))
    return cnn_learner(
        dls,
        arch,
        **{**dict(metrics=[accuracy, top_3_accuracy, top_10_accuracy], cbs=MLflowCallback()), **(learner_kwargs or {})},
    )


def _remove_cbs_of_types(learner: Learner, cb_types: list[type]) -> list[Callback]:
    removed_cbs = [cb for cb in learner.cbs if type(cb) in cb_types]
    learner.remove_cbs(removed_cbs)
    return removed_cbs


def get_learner_metrics_with_tta(learner: Learner, tta_prefix: str = "", **tta_kwargs) -> dict[str, float]:
    """
    Use test-time augmentation (TTA) to calculate the learner's metrics on the validation set.

    :param learner: The learner.
    :param tta_prefix: prefix to set on the TTA metrics. If the default empty string is used, the TTA metrics may
                       overwrite non-TTA metrics.
    :param tta_kwargs: kwargs to pass to `learner.tta()`.
    :return: dict mapping metric names to their values, including TTA metrics
    """
    set_seed(13, reproducible=True)
    metric_values = dict(zip(learner.recorder.metric_names[1:-1], learner.recorder.log[1:-1]))
    with learner.no_bar(), learner.no_logging():
        # Keep the old log because of learner.tta() side effects
        old_log = learner.recorder.log
        # Remove the callback because before_epoch() is called by learner.tta() as a side effect
        removed_cbs = _remove_cbs_of_types(learner, [MLflowCallback])
        preds, targs = learner.tta(**tta_kwargs)
        # Restore the log and callback
        learner.recorder.log = old_log
        learner.add_cbs(removed_cbs)
    for metric in learner.metrics:
        metric_values[f"{tta_prefix}{metric.name}"] = metric.func(preds, targs).item()
    return metric_values


def run_lr_find_experiment(
    learner: Learner,
    num_epochs_between_finds: int,
    num_finds: int,
    suggestion_method: callable,
    show_plot: bool = False,
    disable_mlflow: bool = False,
):
    """
    Run a learning rate finder experiment: Initial fine tuning, then a series of learning rate finds.

    :param learner: The learner to run the experiment on.
    :param num_epochs_between_finds: The number of epochs between learning rate finds.
    :param num_finds: The number of learning rate finds to run.
    :param suggestion_method: The method to use for suggesting the learning rate (one of `SuggestionMethod.*`).
    :param show_plot: If true, show the learning rate finder plot on every find.
    :param disable_mlflow: If true, disable mlflow tracking by not adding back the MLflowCallback after each lr_find().
    """
    if not disable_mlflow:
        mlflow.log_param("num_epochs_between_finds", num_epochs_between_finds)
        mlflow.log_param("num_finds", num_finds)
        mlflow.log_param("suggestion_method", suggestion_method.__name__)

    learner.fine_tune(num_epochs_between_finds)
    for i in range(num_finds):
        learner.remove_cb(MLflowCallback)
        # See https://github.com/fastai/fastai/issues/3239: Need to remove the MixUp callback (and potentially others)
        # to avoid errors.
        removed_cbs = _remove_cbs_of_types(learner, [MixUp])

        suggestions = learner.lr_find(suggest_funcs=[suggestion_method], show_plot=show_plot)
        next_lr = getattr(suggestions, suggestion_method.__name__)
        print(f"Next learning rate: {next_lr:.6f}")

        if not disable_mlflow:
            learner.add_cb(MLflowCallback(run_name_prefix=f"after find {i} - "))
        learner.add_cbs(removed_cbs)

        learner.fit_one_cycle(num_epochs_between_finds, next_lr)
