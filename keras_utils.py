import numpy as np
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from generic_utils import convert_image_to_array


def count_num_features(model=None):
    if model is None:
        model = get_model(input_shape=None)
    num_features = np.product(model.output_shape[1:])
    return num_features

def get_clip_preprocessing():
    return None

def get_model_resolution():
    resolution = 256
    return resolution


def preprocess_image_array_for_model(image_array):
    image_array = preprocess_input(image_array)
    return image_array


def get_model(input_shape, include_top=False, pooling="avg"):
    model = MobileNetV3Small(input_shape=input_shape, include_top=include_top, pooling=pooling)
    return model


def label_image(image, model, preprocess=None):
    # Reference: https://github.com/glouppe/blackbelt/

    # convert the image pixels to a numpy array
    image = convert_image_to_array(image)

    # reshape data for the model
    image = np.expand_dims(image, axis=0)

    # prepare the image for the VGG model
    image = preprocess_image_array_for_model(image)

    # predict the probability across all output classes
    yhat = model.predict(image)

    return yhat
