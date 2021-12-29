# deep-fish
Experimenting with deep learning for fish ID.

## Setup

Create the Conda environment for development:

    $ conda env create --file environment-dev.yml

Install pre-commit hooks:

    $ pre-commit install

Note that the Conda environment for Streamlit in production uses the default `environment.yml` file. This is due to the
following limitations of Streamlit Cloud, which appear to stem from them installing the requirements into an existing
environment rather than creating one from scratch:

* `environment.yml` excludes the `name` field. When it is included, Streamlit doesn't find `fastai`.
* `environment.yml` includes only the minimal dependencies need to run the app, which avoids hitting the error discussed
  [here](https://discuss.streamlit.io/t/error-cannot-uninstall-entrypoints-it-is-a-distutils-installed-project-and-thus-we-cannot-accurately-determine-which-files-belong-to-it-which-would-lead-to-only-a-partial-uninstall-condaenvexception-pip-failed/16708).

## Jupyter notebooks

Run with `PYTHONPATH` to access functionality from the root Python files.

    $ PYTHONPATH=~/projects/deep-fish jupyter notebook

## Fish ID app

Run via streamlit in local development mode (run on save, use local species images, and expose beta features):

    $ streamlit run --server.runOnSave true app.py -- dev

Run via streamlit in production mode:

    $ streamlit run app.py

Build a new model by running the code in `notebooks/03-app.ipynb`.

## Experiment monitoring

Use MLflow:

    $ mlflow ui --backend-store-uri sqlite:///mlruns.db

## Command line interface

Create an RLS species dataset:

    $ python manage.py create-rls-species-dataset \
        --m1-csv-path ~/projects/fish-id/data/dump-20210717/m1.csv \
        --image-dir ~/projects/yanirs.github.io/tools/rls/img \
        --output-dir data/rls-species-25-min-images-3/ \
        --num-species 25 \
        --min-images-per-species 3

Create an RLS genus dataset:

    $ python manage.py create-rls-genus-dataset \
        --image-dir ~/projects/yanirs.github.io/tools/rls/img \
        --output-dir data/rls-top-5-genera \
        --num-top-genera 5

Create a test dataset from a trip directory:

    $ python manage.py create-test-dataset \
        --trip-dir ~/Pictures/202010\ Eviota\ GBR \
        --output-dir data/eviota-202010
