from time import time

import numpy as np
from generic_utils import load_image
from PIL import ImageOps as pil_imageops

from app_id_utils import freeze_app_ids, list_app_ids, app_id_to_image_filename
from data_utils import get_label_database_filename
from model_utils import (
    get_target_model_size,
    load_model,
    convert_image_to_features,
    get_num_features,
    get_preprocessing_tool,
)


def build_feature_index(
    is_horizontal_banner=False, resolution=None, apply_flip=False, apply_mirror=False
):
    pooling = "avg"
    feature_filename = get_label_database_filename(pooling)

    app_ids = list_app_ids(is_horizontal_banner=is_horizontal_banner)
    num_games = len(app_ids)

    target_model_size = get_target_model_size(resolution=resolution)
    model = load_model(target_model_size=target_model_size, pooling=pooling)
    preprocess = get_preprocessing_tool()

    try:
        Y_hat = np.load(feature_filename)
    except FileNotFoundError:
        num_features = get_num_features(model)
        Y_hat = np.zeros((num_games, num_features))

    start = time()

    app_ids = sorted(app_ids, key=int)
    freeze_app_ids(app_ids)

    for (counter, app_id) in enumerate(app_ids):

        # Avoid re-computing values of Y_hat which were previously computed and saved to disk, then recently loaded
        if any(Y_hat[counter, :] != 0):
            continue

        image_filename = app_id_to_image_filename(app_id, is_horizontal_banner)
        image = load_image(image_filename, target_size=target_model_size)
        if apply_flip:
            image = pil_imageops.flip(image)
        if apply_mirror:
            image = pil_imageops.mirror(image)
        features = convert_image_to_features(image, model, preprocess=preprocess)

        Y_hat[counter, :] = features

        if (counter % 1000) == 0:
            print("{}/{} in {:.2f} s".format(counter, num_games, time() - start))
            np.save(feature_filename, Y_hat)

    np.save(feature_filename, Y_hat)

    return


if __name__ == "__main__":
    build_feature_index(
        is_horizontal_banner=False,
        resolution=None,
        apply_flip=False,
        apply_mirror=False,
    )
