from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image
import tensorflow as tf

from config import CLASSES, IMG_SIZE, MODEL_PATH

import keras

_model = None

class CustomDense(keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return super().from_config(config)

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
        with keras.saving.custom_object_scope({'Dense': CustomDense}):
            _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict_from_image(image: Image.Image) -> dict:
    model = get_model()
    batch = preprocess_image(image)
    probabilities = model.predict(batch, verbose=0)[0]

    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])

    all_scores = {
        CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))
    }

    return {
        "label": CLASSES[predicted_idx],
        "confidence": confidence,
        "all_scores": all_scores,
    }


def predict_from_bytes(data: bytes) -> dict:
    image = Image.open(BytesIO(data))
    return predict_from_image(image)


def predict_from_file(file_obj: BinaryIO) -> dict:
    image = Image.open(file_obj)
    return predict_from_image(image)
