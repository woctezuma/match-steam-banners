# from keras_utils import get_model, get_model_resolution, label_image, count_num_features
from openai_utils import get_model, get_model_resolution, label_image, count_num_features


def get_num_features(model=None):
    return count_num_features(model)


def get_target_model_size(resolution=None):
    if resolution is None:
        resolution = get_model_resolution()

    target_model_size = (resolution, resolution)

    return target_model_size


def get_input_shape(target_model_size, num_channels=3):
    # Image data format: channels last
    input_shape = tuple(list(target_model_size) + [num_channels])

    return input_shape


def load_keras_model(target_model_size=None, include_top=False, pooling="avg"):
    # Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v3.py

    if target_model_size is None:
        target_model_size = get_target_model_size()

    input_shape = get_input_shape(target_model_size)
    model = get_model(input_shape=input_shape, include_top=include_top, pooling=pooling)

    return model


def convert_image_to_features(image, model):
    yhat = label_image(image, model)  # runtime: 1 second

    features = yhat.flatten()

    return features
