"""Assorted experiment utilities."""
import contextlib
import typing
from pathlib import Path

import torch

# Fail silently because we don't need mlflow when running only inference.
with contextlib.suppress(ImportError):
    import mlflow
from fastai.callback.core import Callback
from fastai.callback.mixup import MixUp
from fastai.data.block import CategoryBlock, DataBlock
from fastai.data.transforms import RandomSplitter, get_image_files
from fastai.learner import Learner, Recorder
from fastai.metrics import accuracy, top_k_accuracy
from fastai.torch_core import (
    get_random_states,
    set_random_states,
    set_seed,
    show_image,
    tensor,
)
from fastai.vision.augment import RandomResizedCrop, aug_transforms
from fastai.vision.data import ImageBlock
from fastai.vision.learner import vision_learner
from fastcore.basics import range_of
from fastcore.foundation import L as fastcore_list  # noqa: N811


class MLflowCallback(Callback):  # type: ignore[misc]
    """A Learner callback that logs the metrics of each epoch to MLflow."""

    run_after = Recorder

    def __init__(self, start_step: int = 0, **kwargs: typing.Any):
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
    """Split a DataBlock so that all the data is used for training.

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
    """Create a learner that should yield reproducible results across runs.

    Parameters
    ----------
    arch
        the architecture
    dataset_path
        the path of the dataset
    db_kwargs
        optional keyword arguments for the DataBlock
    dls_kwargs
        optional keyword arguments for the DataLoaders
    learner_kwargs
        optional keyword arguments for the Learner

    Returns
    -------
    Learner
        the learner, as produced by `vision_learner()`
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
    return vision_learner(
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
    """Return the learner's metrics on the validation set with test-time augmentation.

    Parameters
    ----------
    learner
        The learner.
    tta_prefix
        prefix to set on the TTA metrics. If the default empty string is
        used, the TTA metrics may overwrite non-TTA metrics.
    **tta_kwargs
        kwargs to pass to `learner.tta()`.

    Returns
    -------
    dict[str, float]
        mapping from metric names to their values, including TTA metrics
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
    """Run a learning rate finder experiment.

    This includes initial fine-tuning, then a series of learning rate finds.

    Parameters
    ----------
    learner
        The learner to run the experiment on.
    num_epochs_between_finds
        The number of epochs between learning rate finds.
    num_finds
        The number of learning rate finds to run.
    suggestion_method
        Method to use for suggesting the learning rate (one of `SuggestionMethod.*`).
    show_plot
        If true, show the learning rate finder plot on every find.
    disable_mlflow
        If true, disable mlflow tracking by not adding back the MLflowCallback after
        each lr_find().
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
    """Test a learner on an unseen set of images, optionally showing some predictions.

    Parameters
    ----------
    learner
        The learner to test.
    image_paths
        The paths to the images to test.
    labels
        The labels for the images.
    show_grid
        The number of rows and columns to visualize; None to show nothing.

    Returns
    -------
    dict[str, float]
        Mapping from top_k_accuracy metric name to its value.
    """
    test_dl = learner.dls.test_dl(image_paths)
    preds = learner.get_preds(dl=test_dl, reorder=False)[0]
    label_codes = tensor([learner.dls.vocab.o2i.get(label, -1) for label in labels])

    if show_grid:
        from matplotlib import pyplot as plt

        for ax, img, label, pred in zip(  # noqa: B905
            plt.subplots(*show_grid, figsize=(14, 16))[1].flatten(),
            test_dl.show_batch(show=False)[0],
            labels,
            preds,
        ):
            show_image(img, ctx=ax)
            pred_label = learner.dls.vocab[pred.argmax()]
            ax.set_title(f"[Actual] {label}\n[Predicted] {pred_label}")
        plt.tight_layout()

    return {
        f"top_{k}_accuracy": top_k_accuracy(preds, label_codes, k).item()
        for k in (1, 3, 10)
    }


class _SaveCheckpointCallback(Callback):  # type: ignore[misc]
    """A simplified version of SaveModelCallback.

    Similarly to SaveModelCallback, this callback saves the model according to the
    every_epoch argument. Unlike SaveModelCallback, it overwrites the latest checkpoint
    and stores all the random states for reproducibility.
    """

    # Values copied from SaveModelCallback.
    # TODO: magic numbers for order are incredibly brittle -- move away from fast.ai
    order = 61
    remove_on_fetch = True
    _only_train_loop = True

    def __init__(self, every_epoch: int, checkpoint_path: Path, min_epoch: int):
        super().__init__()
        self.every_epoch = every_epoch
        self.checkpoint_path = checkpoint_path
        self.min_epoch = min_epoch

    def after_epoch(self) -> None:
        if self.epoch < self.min_epoch or (self.epoch % self.every_epoch):
            return
        torch.save(
            dict(
                epoch=self.epoch,
                model=self.model.state_dict(),
                opt=self.opt.state_dict(),
                random_states=get_random_states(),
            ),
            self.checkpoint_path,
        )
        print(f"Checkpoint saved for epoch {self.epoch}")


def _load_learner_checkpoint(learner: Learner, checkpoint_path: Path) -> int:
    """Load the model & opt state, recover the random state, and return the epoch."""
    if learner.opt is None:
        learner.create_opt()
    checkpoint = torch.load(checkpoint_path)
    learner.model.load_state_dict(checkpoint["model"], strict=True)
    learner.opt.load_state_dict(checkpoint["opt"])  # type: ignore[union-attr]
    set_random_states(**checkpoint["random_states"])
    return checkpoint["epoch"]  # type: ignore[no-any-return]


def resumable_fine_tune(
    learner: Learner,
    model_path: Path,
    epochs: int,
    checkpoint_every_epoch: int = 10,
    base_lr: float = 2e-3,
    freeze_epochs: int = 1,
    lr_mult: float = 100,
    pct_start: float = 0.3,
    div: float = 5.0,
    **kwargs: typing.Any,
) -> None:
    """Inline learner.fine_tune() to support resuming from a model checkpoint.

    Arguments are as for learner.fine_tune() with the addition of model_path. If that
    path exists with the '.ckpt' suffix, then it will be loaded and training will be
    resumed based on the checkpoint data in the file.

    The implementation is based on the example in fastai's 14_callback.schedule.ipynb
    (under _Resume training from checkpoint_). As in the notebook, for this to work,
    this function should be called with the same learner parameters as used for the
    stored checkpoint. This makes it quite brittle, especially when combined with the
    maze of callbacks and delegated attributes.

    TODO: While fine tuning repeatedly from the same restored checkpoint yields
    TODO: reproducible results, it is different from the results obtained by training
    TODO: without checkpoints. It's unclear why and hard to debug. Rather than investing
    TODO: in debugging, it's probably better to drop the fastai dependency and then
    TODO: figure it out with pure PyTorch.
    """
    checkpoint_path = model_path.with_suffix(".ckpt")
    if checkpoint_path.exists():
        start_epoch = _load_learner_checkpoint(learner, checkpoint_path) + 1
    else:
        learner.freeze()
        learner.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
        start_epoch = 0
    # There are two reasons to unfreeze here:
    #  - if we start from a loaded learner that's created with vision_learner(), it
    #    includes a call to freeze(), so we're getting a frozen learner even if we start
    #    from a checkpoint
    #  - if we're fine-tuning from scratch, there's an explicit call to freeze() above
    learner.unfreeze()
    base_lr /= 2
    learner.add_cb(
        _SaveCheckpointCallback(
            checkpoint_every_epoch, checkpoint_path, min_epoch=start_epoch or 1
        )
    )
    learner.fit_one_cycle(
        epochs,
        slice(base_lr / lr_mult, base_lr),
        pct_start=pct_start,
        div=div,
        start_epoch=start_epoch,
        **kwargs,
    )
