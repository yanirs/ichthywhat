import pandas as pd
from fastapi import FastAPI
from fastapi import File, UploadFile
from fastai.learner import load_learner
from io import BytesIO
from fastai.vision.core import PILImage
from pathlib import Path


# TODO: share with streamlit app
DEFAULT_RESOURCES_PATH = Path(__file__).parent.parent / "resources"
api = FastAPI()
_model = load_learner(DEFAULT_RESOURCES_PATH / "model.pkl")


@api.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float]:
    image = PILImage.create(BytesIO(await file.read()))
    preds = _model.predict(image)[2]
    return pd.Series(data=preds, index=_model.dls.vocab).sort_values(ascending=False).to_dict()
