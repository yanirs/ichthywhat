# Ichthy-what? Fishy photo ID

Experimenting with deep learning for fish identification with Reef Life Survey data.

## Setup

Prerequisites: Set up Python 3.10 (e.g., with [pyenv](https://github.com/pyenv/pyenv)) and [Poetry](https://python-poetry.org/).

Set up the Poetry environment:

    $ poetry install

Install pre-commit hooks:

    $ poetry run pre-commit install

Alternatively, install [Vagrant](https://www.vagrantup.com/) and run everything in a virtual machine:

    $ vagrant up

## Fish ID Streamlit app

Run via streamlit in local development mode (run on save, use local species images, and expose beta features &ndash; the
`ichthywhat.localhost` address is needed for the Mapbox API to work and a mapping should exist in `/etc/hosts`):

    $ poetry run streamlit run --browser.serverAddress ichthywhat.localhost --server.runOnSave true ichthywhat/app.py -- dev /path/to/img/root

Run via streamlit in production mode:

    $ poetry run streamlit run ichthywhat/app.py

Build a new model by running the code in `notebooks/03-app.ipynb`.

## Fish ID classification API

The Vagrant machine exposes a simple classification API. With the machine running, call:

    $ curl -X POST -F "img_file=@image-file-path.jpg" http://localhost:9300/predict | jq

Alternatively, visit http://localhost:9300/demo for a simple demo page.

This API is also packaged in a Dockerfile, which can be built on the Vagrant machine:

    $ podman build -t ichthywhat .

...and run with the default port exposed to the host:

    $ podman run --env UVICORN_HOST=0.0.0.0 -p 8000:8000 localhost/ichthywhat:latest

...and exported elsewhere:

    $ podman save localhost/ichthywhat:latest | gzip > ichthywhat-img.tar.gz

...then on another machine that has Docker, perhaps with a local proxy:

    $ docker load --input ichthywhat-img.tar.gz
    $ docker run -p 127.0.0.1:8000:8000 localhost/ichthywhat:latest

See the ARG and ENV calls in the Dockerfile for customisation options.

## Jupyter notebooks used for experimentation and model building

    $ poetry run jupyter notebook

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
