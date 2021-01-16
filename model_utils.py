# from keras_utils import get_model, get_model_resolution, label_image, count_num_features, get_clip_preprocessing
from openai_utils import get_model, get_model_resolution, label_image, count_num_features, get_clip_preprocessing


def get_num_features(model=None):
    return count_num_features(model)


def get_preprocessing_tool():
    return get_clip_preprocessing()


def get_target_model_size(resolution=None):
    if resolution is None:
        resolution = get_model_resolution()

    target_model_size = (resolution, resolution)

    return target_model_size


def get_input_shape(target_model_size, num_channels=3):
    # Image data format: channels last
    input_shape = tuple(list(target_model_size) + [num_channels])

    return input_shape


def load_model(target_model_size=None, include_top=False, pooling="avg"):
    if target_model_size is None:
        target_model_size = get_target_model_size()

    input_shape = get_input_shape(target_model_size)
    model = get_model(input_shape=input_shape, include_top=include_top, pooling=pooling)

    return model


def convert_image_to_features(image, model, preprocess=None):
    yhat = label_image(image, model, preprocess=preprocess)

    features = yhat.flatten()

    return features
