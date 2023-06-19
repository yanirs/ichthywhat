"""Basic classification API."""
import os
from collections import OrderedDict
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import HTMLResponse

from ichthywhat.constants import DEFAULT_RESOURCES_PATH
from ichthywhat.inference import OnnxWrapper

api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ICHTHYWHAT_API_ALLOW_ORIGINS", "").split(",") or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_onnx_wrapper: None | OnnxWrapper = None


_DEMO_HTML = """
  <!DOCTYPE html>
  <html>
  <head>
    <title>Ichthywhat: Prediction API Demo</title>
  </head>
  <body>
    <h1>Ichthywhat: Prediction API Demo</h1>

    <label for="api-url">API URL</label>
    <input type="url" id="api-url" value="/predict"><br>

    <label for="file-input">Image</label>
    <input type="file" id="file-input">
    <button type="button" onclick="uploadImage()">Upload</button>

    <pre id="output"></pre>

    <script>
      function updateOutput(text) {
        document.getElementById('output').innerHTML = text;
      }

      function uploadImage() {
        updateOutput('Loading...');
        const formData = new FormData();
        formData.append('img_file', document.getElementById('file-input').files[0]);
        fetch(document.getElementById('api-url').value, {
          method: 'POST',
          body: formData
        })
        .then(response => response.json())
        .then(data => updateOutput(JSON.stringify(data, null, 2)))
        .catch(error => updateOutput(error));
      }
    </script>
  </body>
  </html>
"""


@api.get("/")
async def home() -> str:
    """Trivial homepage that serves as a health check."""
    return "Hello!"


@api.get("/demo")
async def demo() -> HTMLResponse:
    """Very basic demo page for image upload via the API."""
    return HTMLResponse(content=_DEMO_HTML)


@api.post("/predict")
async def predict(
    img_file: UploadFile = File(...),  # noqa: B008
) -> OrderedDict[str, float]:
    """Return a mapping from species name to score, based on the given image."""
    global _onnx_wrapper
    if not _onnx_wrapper:
        _onnx_wrapper = OnnxWrapper(DEFAULT_RESOURCES_PATH / "model.onnx")
    img = Image.open(BytesIO(await img_file.read()))
    return _onnx_wrapper.predict(img)
