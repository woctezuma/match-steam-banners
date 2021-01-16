import numpy as np
from keras_utils import get_model, preprocess_image_array_for_model
from generic_utils import convert_image_to_array


def get_target_model_size():
    target_model_size = (256, 256)

    return target_model_size


def load_keras_model(target_model_size=None, include_top=False, pooling="avg"):
    # Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v3.py

    if target_model_size is None:
        target_model_size = get_target_model_size()

    num_channels = 3

    # Image data format: channels last
    input_shape = tuple(list(target_model_size) + [num_channels])

    model = get_model(input_shape=input_shape, include_top=include_top, pooling=pooling)

    return model


def label_image(image, model):
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


def convert_image_to_features(image, model):
    yhat = label_image(image, model)  # runtime: 1 second

    features = yhat.flatten()

    return features
