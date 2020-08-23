from pathlib import Path

import numpy as np
from keras.preprocessing.image import load_img
from sklearn.neighbors import NearestNeighbors

from app_id_utils import app_id_to_image_filename, get_frozen_app_ids
from data_utils import get_label_database_filename
from download_utils import download_query_image
from model_utils import (
    get_target_model_size,
    load_keras_model,
    convert_image_to_features,
)
from print_utils import print_ranking_for_app_id
from steam_spy_utils import load_benchmarked_app_ids, load_game_names_from_steamspy


def retrieve_similar_features(
    query_app_id,
    knn,
    target_model_size,
    keras_model,
    is_horizontal_banner=False,
    num_neighbors=10,
):
    image_filename = app_id_to_image_filename(query_app_id, is_horizontal_banner)
    if not Path(image_filename).is_file():
        download_query_image(
            query_app_id,
            is_horizontal_banner=is_horizontal_banner,
            output_filename=image_filename,
        )
    image = load_img(image_filename, target_size=target_model_size)
    query_des = convert_image_to_features(image, keras_model)

    # Sci-Kit Learn with cosine similarity. Reshape data as it contains a single sample.
    _, matches = knn.kneighbors(query_des.reshape(1, -1), n_neighbors=num_neighbors)

    app_ids = get_frozen_app_ids()

    # When we use the Keras model, a Steam banner is represented by only ONE feature, hence the use of 'matches[0]'.
    reference_app_id_counter = [app_ids[element] for element in matches[0]]

    return reference_app_id_counter


def get_knn_search_structure(label_database, use_cosine_similarity=True):
    if use_cosine_similarity:
        knn = NearestNeighbors(metric="cosine", algorithm="brute")
        knn.fit(label_database)
    else:
        knn = NearestNeighbors(algorithm="brute")
        knn.fit(label_database)

    return knn


def batch_retrieve_similar_features(
    query_app_ids=None,
    use_cosine_similarity=True,
    is_horizontal_banner=False,
    pooling="avg",
    num_neighbors=10,
):
    if query_app_ids is None:
        query_app_ids = load_benchmarked_app_ids()

    label_database = np.load(get_label_database_filename(pooling))
    knn = get_knn_search_structure(label_database, use_cosine_similarity)

    target_model_size = get_target_model_size()
    keras_model = load_keras_model(target_model_size=target_model_size, pooling=pooling)

    game_names = load_game_names_from_steamspy()

    for query_app_id in query_app_ids:

        try:
            reference_app_id_counter = retrieve_similar_features(
                query_app_id,
                knn,
                target_model_size,
                keras_model,
                is_horizontal_banner=is_horizontal_banner,
                num_neighbors=num_neighbors,
            )
        except FileNotFoundError:
            print("Query image not found for appID={}.".format(query_app_id))
            continue

        print_ranking_for_app_id(
            query_app_id, reference_app_id_counter, game_names=game_names,
        )

    return


if __name__ == "__main__":
    query_app_ids = load_benchmarked_app_ids()
    batch_retrieve_similar_features(query_app_ids, is_horizontal_banner=False)
