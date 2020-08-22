from time import time

import numpy as np
from keras.preprocessing.image import load_img

from app_id_utils import freeze_app_ids, list_app_ids, app_id_to_image_filename
from data_utils import get_label_database_filename
from model_utils import (
    get_target_model_size,
    load_keras_model,
    convert_image_to_features,
)


def build_feature_index(is_horizontal_banner=False):
    pooling = "avg"
    feature_filename = get_label_database_filename(pooling)

    app_ids = list_app_ids(is_horizontal_banner=is_horizontal_banner)
    num_games = len(app_ids)

    target_model_size = get_target_model_size()
    model = load_keras_model(target_model_size=target_model_size, pooling=pooling)

    try:
        Y_hat = np.load(feature_filename)
    except FileNotFoundError:
        num_features = np.product(model.output_shape[1:])
        Y_hat = np.zeros((num_games, num_features))

    start = time()

    for (counter, app_id) in enumerate(sorted(app_ids, key=int)):

        # Avoid re-computing values of Y_hat which were previously computed and saved to disk, then recently loaded
        if any(Y_hat[counter, :] != 0):
            continue

        image_filename = app_id_to_image_filename(app_id, is_horizontal_banner)
        image = load_img(image_filename, target_size=target_model_size)
        features = convert_image_to_features(image, model)

        Y_hat[counter, :] = features

        if (counter % 1000) == 0:
            print("{}/{} in {:.2f} s".format(counter, num_games, time() - start))

            if Y_hat is not None:
                saving_start = time()
                np.save(feature_filename, Y_hat)
                freeze_app_ids(app_ids)
                print("Elapsed time (saving): {:.2f} s".format(time() - saving_start))

    np.save(feature_filename, Y_hat)
    freeze_app_ids(app_ids)

    return


if __name__ == "__main__":
    build_feature_index(is_horizontal_banner=False)
