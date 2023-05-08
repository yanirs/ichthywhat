"""Assorted experiment utilities."""
import contextlib
import typing
from pathlib import Path

# Fail silently because we don't need mlflow when running only inference.
with contextlib.suppress(ImportError):
    import mlflow
from fastai.callback.core import Callback
from fastai.callback.mixup import MixUp
from fastai.data.block import CategoryBlock, DataBlock
from fastai.data.transforms import RandomSplitter, get_image_files
from fastai.learner import Learner, Recorder
from fastai.metrics import accuracy, top_k_accuracy
from fastai.torch_core import set_seed, show_image, tensor
from fastai.vision.augment import RandomResizedCrop, aug_transforms
from fastai.vision.data import ImageBlock
from fastai.vision.learner import cnn_learner
from fastcore.basics import range_of
from fastcore.foundation import L as fastcore_list  # noqa: N811


class MLflowCallback(Callback):  # type: ignore[misc]
    """A Learner callback that logs the metrics of each epoch to MLflow."""

    run_after = Recorder

    def __init__(self, start_step: int = 0, **kwargs: typing.Any):  # noqa: D107
        super().__init__(**kwargs)
        self.step = start_step

    def after_epoch(self) -> None:  # noqa: D102
        self.step += 1
        mlflow.log_metrics(
            dict(
                zip(
                    self.recorder.metric_names[1:-1], self.recorder.log[1:], strict=True
                )
            ),
            step=self.step,
        )


def top_3_accuracy(inp: typing.Sequence[float], targ: typing.Sequence[float]) -> float:
    """Delegate to top_k_accuarcy with k=3."""
    return top_k_accuracy(inp, targ, k=3)  # type: ignore[no-any-return]


def top_10_accuracy(inp: typing.Sequence[float], targ: typing.Sequence[float]) -> float:
    """Delegate to top_k_accuarcy with k=10."""
    return top_k_accuracy(inp, targ, k=10)  # type: ignore[no-any-return]


def delete_run_with_children(parent_run_id: str) -> None:
    """Delete an MLflow run together with all its children."""
    client = mlflow.tracking.client.MlflowClient()
    parent_run = client.get_run(parent_run_id)
    for child_run in client.search_runs(
        experiment_ids=[parent_run.info.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
    ):
        client.delete_run(child_run.info.run_id)
    client.delete_run(parent_run_id)


def get_species_from_path(path: Path) -> str:
    """Get the species name from path, assuming that it's named `<genus>-<taxon>-.*`."""
    return " ".join(path.name.split("-")[:2]).capitalize()


def no_validation_splitter(items: typing.Sequence[typing.Any]) -> fastcore_list:
    """
    Split a DataBlock so that all the data is used for training.

    See notebooks/03-app.ipynb for a usage example.
    """
    return fastcore_list(range_of(items)), fastcore_list([])


def create_reproducible_learner(
    arch: typing.Any,
    dataset_path: Path,
    db_kwargs: dict[str, typing.Any] | None = None,
    dls_kwargs: dict[str, typing.Any] | None = None,
    learner_kwargs: dict[str, typing.Any] | None = None,
) -> Learner:
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
        **{
            **dict(
                metrics=[accuracy, top_3_accuracy, top_10_accuracy],
                cbs=MLflowCallback(),
            ),
            **(learner_kwargs or {}),
        },
    )


def _remove_cbs_of_types(learner: Learner, cb_types: list[type]) -> list[Callback]:
    removed_cbs = [cb for cb in learner.cbs if type(cb) in cb_types]
    learner.remove_cbs(removed_cbs)
    return removed_cbs


def get_learner_metrics_with_tta(
    learner: Learner, tta_prefix: str = "", **tta_kwargs: dict[str, typing.Any]
) -> dict[str, float]:
    """
    Use test-time augmentation to calculate the learner's metrics on the validation set.

    :param learner: The learner.
    :param tta_prefix: prefix to set on the TTA metrics. If the default empty string is
                       used, the TTA metrics may overwrite non-TTA metrics.
    :param tta_kwargs: kwargs to pass to `learner.tta()`.
    :return: dict mapping metric names to their values, including TTA metrics
    """
    set_seed(13, reproducible=True)
    metric_values = dict(
        zip(
            learner.recorder.metric_names[1:-1], learner.recorder.log[1:-1], strict=True
        )
    )
    with learner.no_bar(), learner.no_logging():
        # Temporarily remove the logging callbacks because before_epoch() is called by
        # learner.tta() as a side effect.
        removed_cbs = _remove_cbs_of_types(learner, [MLflowCallback, Recorder])
        preds, targs = learner.tta(**tta_kwargs)
        learner.add_cbs(removed_cbs)
    for metric in learner.metrics:
        metric_values[f"{tta_prefix}{metric.name}"] = metric.func(preds, targs).item()
    return metric_values


def run_lr_find_experiment(
    learner: Learner,
    num_epochs_between_finds: int,
    num_finds: int,
    suggestion_method: typing.Callable,  # type: ignore[type-arg]
    show_plot: bool = False,
    disable_mlflow: bool = False,
) -> None:
    """
    Run a learning rate finder experiment.

    This includes initial fine tuning, then a series of learning rate finds.

    :param learner: The learner to run the experiment on.
    :param num_epochs_between_finds: The number of epochs between learning rate finds.
    :param num_finds: The number of learning rate finds to run.
    :param suggestion_method: The method to use for suggesting the learning rate
                              (one of `SuggestionMethod.*`).
    :param show_plot: If true, show the learning rate finder plot on every find.
    :param disable_mlflow: If true, disable mlflow tracking by not adding back the
                           MLflowCallback after each lr_find().
    """
    if not disable_mlflow:
        mlflow.log_param("num_epochs_between_finds", num_epochs_between_finds)
        mlflow.log_param("num_finds", num_finds)
        mlflow.log_param("suggestion_method", suggestion_method.__name__)

    learner.fine_tune(num_epochs_between_finds)
    for i in range(num_finds):
        learner.remove_cb(MLflowCallback)
        # See https://github.com/fastai/fastai/issues/3239: Need to remove the MixUp
        # callback (and potentially others) to avoid errors.
        removed_cbs = _remove_cbs_of_types(learner, [MixUp])

        suggestions = learner.lr_find(
            suggest_funcs=[suggestion_method], show_plot=show_plot
        )
        next_lr = getattr(suggestions, suggestion_method.__name__)
        print(f"Next learning rate: {next_lr:.6f}")

        if not disable_mlflow:
            learner.add_cb(MLflowCallback(run_name_prefix=f"after find {i} - "))
        learner.add_cbs(removed_cbs)

        # TODO: This may be wrong -- we don't actually use the next_lr in all iterations
        # TODO: because fit_one_cycle() has a cyclical learning rate. It may be better
        # TODO: to use fit() directly. It's also wrong to remove the MixUp callback
        # TODO: because then we get loss estimates that ignore its effect.
        learner.fit_one_cycle(num_epochs_between_finds, next_lr)


def test_learner(
    learner: Learner,
    image_paths: typing.Sequence[Path],
    labels: typing.Sequence[str],
    show_grid: tuple[int, int] = (4, 4),
) -> dict[str, float]:
    """
    Test a learner on an unseen set of images, optionally showing some predictions.

    :param learner: The learner to test.
    :param image_paths: The paths to the images to test.
    :param labels: The labels for the images.
    :param show_grid: The number of rows and columns to visualize; None to show nothing.

    :return: Mapping from top_k_accuracy metric name to its value.
    """
    test_dl = learner.dls.test_dl(image_paths)
    preds = learner.get_preds(dl=test_dl, reorder=False)[0]
    label_codes = tensor([learner.dls.vocab.o2i.get(label, -1) for label in labels])

    if show_grid:
        from matplotlib import pyplot as plt

        for ax, img, label, pred in zip(
            plt.subplots(*show_grid, figsize=(14, 16))[1].flatten(),
            test_dl.show_batch(show=False)[0],
            labels,
            preds,
            strict=True,
        ):
            show_image(img, ctx=ax)
            pred_label = learner.dls.vocab[pred.argmax()]
            ax.set_title(f"[Actual] {label}\n[Predicted] {pred_label}")
        plt.tight_layout()

    return {
        f"top_{k}_accuracy": top_k_accuracy(preds, label_codes, k).item()
        for k in (1, 3, 10)
    }
