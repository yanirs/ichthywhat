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
    """Return a mapping from species name to score, based on the given image."""
    image = PILImage.create(BytesIO(await img_file.read()))
    scores = _model.predict(image)[2]
    return pd.Series(data=scores, index=_model.dls.vocab).sort_values(ascending=False).to_dict()
