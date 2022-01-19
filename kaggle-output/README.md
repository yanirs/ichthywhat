## Example: Importing `mlruns` from Kaggle

Needed because the version saved by Kaggle is for the latest run.

    $ kaggle kernels output yanirseroussi/deep-fish-initial-experiments
    $ mv deep-fish/mlruns.db deep-fish-initial-experiments/mlruns-v12.db
    $ rm <superfluous files>

TODO: merge the remote database into the root `mlruns.db`. This would probably require a database-level merge given the
limitations of `mlflow-export-import` listed in https://github.com/mlflow/mlflow/issues/2382#issuecomment-864051446.
