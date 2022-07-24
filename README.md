# deep-fish
Experimenting with deep learning for fish ID.

## Setup

Prerequisites: Set up Python 3.9 (e.g., with [pyenv](https://github.com/pyenv/pyenv)) and [Poetry](https://python-poetry.org/).

Set up the Poetry environment:

    $ poetry install

Install pre-commit hooks:

    $ poetry run pre-commit install

## Jupyter notebooks

Run with `PYTHONPATH` to access functionality from the root Python files.

    $ PYTHONPATH=/path/to/deep-fish poetry run jupyter notebook

## Fish ID app

Run via streamlit in local development mode (run on save, use local species images, and expose beta features &ndash; the
`ichthywhat.localhost` address is needed for the Mapbox API to work and a mapping should exist in `/etc/hosts`):

    $ poetry run streamlit run --browser.serverAddress ichthywhat.localhost --server.runOnSave true app.py -- dev /path/to/img/root

Run via streamlit in production mode:

    $ poetry run streamlit run app.py

Build a new model by running the code in `notebooks/03-app.ipynb`.

## Experiment monitoring

Use MLflow:

    $ poetry run mlflow ui --backend-store-uri sqlite:///mlruns.db

## Command line interface

Create an RLS species dataset:

    $ poetry run ichthywhat create-rls-species-dataset \
        --m1-csv-path ~/projects/fish-id/data/dump-20210717/m1.csv \
        --image-dir ~/projects/yanirs.github.io/tools/rls/img \
        --output-dir data/rls-species-25-min-images-3/ \
        --num-species 25 \
        --min-images-per-species 3

Create an RLS genus dataset:

    $ poetry run ichthywhat create-rls-genus-dataset \
        --image-dir ~/projects/yanirs.github.io/tools/rls/img \
        --output-dir data/rls-top-5-genera \
        --num-top-genera 5

Create a test dataset from a trip directory:

    $ poetry run ichthywhat create-test-dataset \
        --trip-dir ~/Pictures/202010\ Eviota\ GBR \
        --output-dir data/eviota-202010
