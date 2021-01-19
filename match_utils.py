# Objective: find **the** nearest neighbor of each feature vector (encoding a banner) in the database.

import numpy as np

from app_id_utils import get_frozen_app_ids
from build_feature_index import get_label_database_filename
from data_utils import save_sim_dict
from knn_utils import get_knn_search_structure, find_knn_for_all


def match_all(use_cosine_similarity=True, pooling="avg"):
    label_database = np.load(get_label_database_filename(pooling))
    knn = get_knn_search_structure(label_database, use_cosine_similarity)

    query = None
    num_neighbors = 1

    # Caveat: the output is the cosine distance, not the cosine similarity! Transform it before using it!
    dist, matches = find_knn_for_all(knn, query, num_neighbors)

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
