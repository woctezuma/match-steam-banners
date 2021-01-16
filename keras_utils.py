import numpy as np
from PIL import Image as pil_image
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


def get_interpolation_methods():
    interpolation_methods = {
        "nearest": pil_image.NEAREST,
        "bilinear": pil_image.BILINEAR,
        "bicubic": pil_image.BICUBIC,
    }

    return interpolation_methods


def load_image(image_filename, color_mode="RGB", target_size=None, interpolation="nearest"):
    interpolation_methods = get_interpolation_methods()
    resample = interpolation_methods[interpolation]

    # Ensure that letters are ALL upper-case, e.g. 'RGB' instead of 'rgb'
    color_mode = color_mode.upper()

    with pil_image.open(image_filename, "r") as img:
        if img.mode != color_mode:
            img = img.convert(color_mode)

        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                img = img.resize(width_height_tuple, resample)

    return img


def convert_image_to_array(img, dtype="float32"):
    x = np.asarray(img, dtype=dtype)
    return x


def preprocess_image_array_for_model(image_array):
    image_array = preprocess_input(image_array)
    return image_array


def get_model(input_shape, include_top, pooling):
    model = MobileNetV3Small(input_shape=input_shape, include_top=include_top, pooling=pooling)
    return model
