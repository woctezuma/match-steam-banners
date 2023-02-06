from time import time

import numpy as np
from PIL import ImageOps as pil_imageops

from app_id_utils import app_id_to_image_filename, freeze_app_ids, list_app_ids
from data_utils import get_label_database_filename
from generic_utils import load_image
from model_utils import (
    convert_image_to_features,
    get_num_features,
    get_preprocessing_tool,
    get_target_model_size,
    load_model,
)


def build_feature_index(
    is_horizontal_banner=False,
    resolution=None,
    apply_flip=False,
    apply_mirror=False,
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

    for counter, app_id in enumerate(app_ids):
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
            print(f"{counter}/{num_games} in {time() - start:.2f} s")
            np.save(
                feature_filename,
                np.asarray(Y_hat, dtype=np.float16),
                allow_pickle=False,
                fix_imports=False,
            )

    np.save(
        feature_filename,
        np.asarray(Y_hat, dtype=np.float16),
        allow_pickle=False,
        fix_imports=False,
    )

    return


if __name__ == "__main__":
    build_feature_index(
        is_horizontal_banner=False,
        resolution=None,
        apply_flip=False,
        apply_mirror=False,
    )
