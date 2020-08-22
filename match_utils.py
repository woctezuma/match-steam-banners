from time import time

import numpy as np

from app_id_utils import get_frozen_app_ids
from build_feature_index import get_label_database_filename
from data_utils import save_sim_dict
from retrieve_similar_features import get_knn_search_structure


def match_all(use_cosine_similarity=True, pooling="avg"):
    label_database = np.load(get_label_database_filename(pooling))
    knn = get_knn_search_structure(label_database, use_cosine_similarity)

    query = None
    num_neighbors = 1

    # Caveat: the output 'dist' returned by knn.kneighbors() is the 'cosine distance', not the cosine similarity!
    # Reference: https://en.wikipedia.org/wiki/Cosine_similarity

    start = time()
    dist, matches = knn.kneighbors(X=query, n_neighbors=num_neighbors)
    print("Elapsed time: {:.2f} s".format(time() - start))

    app_ids = get_frozen_app_ids()

    sim_dict = dict()
    for counter, query_app_id in enumerate(app_ids):
        last_index = num_neighbors - 1

        second_best_match = matches[counter][last_index]
        second_best_matched_app_id = app_ids[second_best_match]

        cosine_distance = dist[counter][last_index]
        second_best_similarity_score = 1.0 - cosine_distance

        sim_dict[query_app_id] = dict()
        sim_dict[query_app_id]["app_id"] = second_best_matched_app_id
        sim_dict[query_app_id]["similarity"] = second_best_similarity_score

    save_sim_dict(sim_dict, pooling=pooling)

    return sim_dict
