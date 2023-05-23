"""ONNX export and inference proof-of-concept."""
# TODO: break this up to modules -- export in models.py, inference elsewhere
# TODO: validate exported onnx model on the QUT dataset
# TODO: refactor streamlit app (use functions) and change it to use ONNX
# TODO: use ONNX in the API (rewrite Dockerfile accordingly)
# TODO: try running inference in the browser
# TODO: check the resizing (different for v2? take centre crop rather than using image.resize?)
# TODO: try to get the normalise transform args from the learner
# TODO: inspired by https://community.wandb.ai/t/taking-fastai-to-production/1705 -- reread to ensure it makes sense

import json

import numpy as np
import onnxruntime as ort
import torch
import torchvision
import torchvision.transforms
from PIL import Image


def onnx_path_to_labels(onnx_path):
    return onnx_path.parent / f"{onnx_path.name}.labels.json"


def export_to_onnx(learner, export_path):
    model = torch.nn.Sequential(
        # Replace fastai's normalisation. Assumes the base model was trained on ImageNet.
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        # Actual PyTorch model.
        learner.model.eval(),
        # Replace fastai's softmax layer.
        torch.nn.Softmax(dim=1),
    )
    torch.onnx.export(
        model,
        # ImageNet default input of 224x224
        # TODO: needs adjustment for v2?
        torch.randn(1, 3, 224, 224),
        export_path,
        input_names=["input"],
        output_names=["output"],
        # Variable length axes.
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    with onnx_path_to_labels(export_path).open("w") as fp:
        json.dump(list(learner.dls.vocab), fp)


def image_transform_onnx(path: str, size: int) -> np.ndarray:
    """Image transform helper for onnx runtime inference."""
    image = Image.open(path)
    image = image.resize((size, size))

    image = np.array(image)

    # Match the exported model input: torch.randn(1, 3, 224, 224)
    image = image.transpose(2, 0, 1).astype(np.float32)

    image /= 255
    image = image[None, ...]
    return image


def predict(onnx_model_path, img_path, target_size=224):
    with onnx_path_to_labels(onnx_model_path).open() as fp:
        labels = json.load(fp)
    img = image_transform_onnx(img_path, target_size)
    sess = ort.InferenceSession(str(onnx_model_path))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_dims = sess.get_inputs()[0].shape
    print(
        f"Input layer name: {input_name}, Output layer name: {output_name}, Input Dimension: {input_dims}"
    )
    results = sess.run([output_name], {input_name: img})[0]
    print(results)
    print(f"It's a {labels[np.argmax(results)]}", results)
