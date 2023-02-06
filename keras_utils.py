import numpy as np
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from generic_utils import convert_image_to_array


def get_model_slug_for_keras():
    return "keras"


def count_num_features_for_keras(model=None):
    if model is None:
        model = get_model_for_keras(input_shape=None)
    num_features = np.product(model.output_shape[1:])
    return num_features


def get_dummy_preprocessing_for_keras():
    return None


def get_model_resolution_for_keras(model=None):
    resolution = 256
    return resolution


def preprocess_image_array_for_model_for_keras(image_array):
    image_array = preprocess_input(image_array)
    return image_array


def get_model_for_keras(input_shape, include_top=False, pooling="avg"):
    # Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v3.py

    model = MobileNetV3Small(
        input_shape=input_shape,
        include_top=include_top,
        pooling=pooling,
    )
    return model


def label_image_for_keras(image, model, preprocess=None):
    # Reference: https://github.com/glouppe/blackbelt/

    # convert the image pixels to a numpy array
    image = convert_image_to_array(image)

    # reshape data for the model
    image = np.expand_dims(image, axis=0)

    # prepare the image for the VGG model
    image = preprocess_image_array_for_model_for_keras(image)

    # predict the probability across all output classes
    yhat = model.predict(image)

    return yhat


if __name__ == "__main__":
    slug_name = get_model_slug_for_keras()
    print("Slug: {}".format(slug_name))
