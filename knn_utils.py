from time import time

from sklearn.neighbors import NearestNeighbors


def get_knn_search_structure(label_database, use_cosine_similarity=True):
    if use_cosine_similarity:
        knn = NearestNeighbors(metric="cosine", algorithm="brute")
        knn.fit(label_database)
    else:
        knn = NearestNeighbors(algorithm="brute")
        knn.fit(label_database)

    return knn


def find_knn_for_a_single_query(knn, query_des, num_neighbors):
    # Sci-Kit Learn with cosine similarity. Reshape data as it contains a single sample.
    dist, matches = knn.kneighbors(query_des.reshape(1, -1), n_neighbors=num_neighbors)

    return dist, matches


def find_knn_for_all(knn, query, num_neighbors):
    # Caveat: the output 'dist' returned by knn.kneighbors() is the 'cosine distance', not the cosine similarity!
    # Reference: https://en.wikipedia.org/wiki/Cosine_similarity

    start = time()
    dist, matches = knn.kneighbors(X=query, n_neighbors=num_neighbors)
    print(f"Elapsed time: {time() - start:.2f} s")

    return dist, matches
