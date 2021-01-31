# Objective: find **the** nearest neighbor of each feature vector (encoding a banner) in the database.

import numpy as np

from app_id_utils import get_frozen_app_ids
from build_feature_index import get_label_database_filename
from data_utils import save_sim_dict
from faiss_utils import get_faiss_search_structure, find_faiss_knn_for_all


def match_all(use_cosine_similarity=True, pooling="avg", transform_distance=False):
    label_database = np.load(get_label_database_filename(pooling))
    knn = get_faiss_search_structure(
        embeddings=label_database, use_cosine_similarity=use_cosine_similarity
    )

    query = None
    num_neighbors = 1

    # Caveat:
    #
    # - with find_knn_for_all(), the output is the cosine distance, not the cosine similarity!
    #   Transform the output before using it! Ensure transform_distance is True!
    #
    # - with find_faiss_knn_for_all(), the output is directly the cosine similarity!
    #   Ensure transform_distance is False!
    dist, matches = find_faiss_knn_for_all(
        index=knn,
        embeddings_for_query=query,
        num_neighbors=num_neighbors,
        use_cosine_similarity=use_cosine_similarity,
    )

    app_ids = get_frozen_app_ids()

    sim_dict = dict()
    for counter, query_app_id in enumerate(app_ids):
        last_index = num_neighbors - 1

        second_best_match = matches[counter][last_index]
        second_best_matched_app_id = app_ids[second_best_match]

        cosine_distance = dist[counter][last_index]
        if transform_distance:
            # If find_knn_for_all() was used, then transform the output as follows:
            second_best_similarity_score = 1.0 - cosine_distance
        else:
            # If find_faiss_knn_for_all() was used, then directly use the output as follows:
            second_best_similarity_score = cosine_distance

        sim_dict[query_app_id] = dict()
        sim_dict[query_app_id]["app_id"] = second_best_matched_app_id
        sim_dict[query_app_id]["similarity"] = second_best_similarity_score

    save_sim_dict(sim_dict, pooling=pooling)

    return sim_dict
