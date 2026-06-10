import tensorflow as tf
from config import MODEL_PATH
import keras

class CustomDense(keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return super().from_config(config)

print(f"Loading model from {MODEL_PATH}")

try:
    # Try using custom object
    with keras.saving.custom_object_scope({'Dense': CustomDense}):
        model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully with custom Dense!")
except Exception as e:
    print("Failed with custom Dense:", e)
