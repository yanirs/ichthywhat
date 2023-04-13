"""Basic classification API."""
from io import BytesIO

import pandas as pd
from fastai.learner import load_learner
from fastai.vision.core import PILImage
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ichthywhat.constants import DEFAULT_RESOURCES_PATH

api = FastAPI()
# TODO: make this more restrictive
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_model = load_learner(DEFAULT_RESOURCES_PATH / "model.pkl")


@api.get("/")
async def home() -> str:
    """Trivial homepage that serves as a health check."""
    return "Hello!"


@api.post("/predict")
async def predict(img_file: UploadFile = File(...)) -> dict[str, float]:  # noqa: B008
    """Return a mapping from species name to score, based on the given image."""
    image = PILImage.create(BytesIO(await img_file.read()))
    scores = _model.predict(image)[2]
    return pd.Series(data=scores, index=_model.dls.vocab).sort_values(ascending=False).to_dict()
