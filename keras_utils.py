from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


def get_model_resolution():
    resolution = 256
    return resolution


def preprocess_image_array_for_model(image_array):
    image_array = preprocess_input(image_array)
    return image_array


def get_model(input_shape, include_top=False, pooling="avg"):
    model = MobileNetV3Small(input_shape=input_shape, include_top=include_top, pooling=pooling)
    return model
