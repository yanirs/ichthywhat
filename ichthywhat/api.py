"""Basic classification API."""
from io import BytesIO

import pandas as pd
from fastai.learner import load_learner
from fastai.vision.core import PILImage
from fastapi import FastAPI, File, UploadFile

from ichthywhat.constants import DEFAULT_RESOURCES_PATH

api = FastAPI()
_model = load_learner(DEFAULT_RESOURCES_PATH / "model.pkl")


@api.post("/predict")
async def predict(img_file: UploadFile = File(...)) -> dict[str, float]:  # noqa: B008
    """
    Return a mapping from species name to score.

    When running with Vagrant, this can be tested with: $ curl -X POST -F "file=@data/demo/P6204455.JPG"
    http://localhost:9300/predict | jq
    """
    image = PILImage.create(BytesIO(await img_file.read()))
    scores = _model.predict(image)[2]
    return pd.Series(data=scores, index=_model.dls.vocab).sort_values(ascending=False).to_dict()
