from time import time

import faiss


def get_faiss_search_structure(embeddings, use_cosine_similarity=True):
    d = embeddings.shape[1]
    xb = embeddings.astype('float32')

    if use_cosine_similarity:
        # Caveat: you need to normalize the embeddings,
        #         because faiss uses dot-product instead of cosine similarity!
        # cf. https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#metric_inner_product
        faiss.normalize_L2(xb)

        index = faiss.IndexFlatIP(d)
        index.add(xb)
    else:
        index = faiss.IndexFlatL2(d)
        index.add(xb)

    return index


def find_knn_for_all(index, embeddings_for_query, num_neighbors, use_cosine_similarity=True):
    xq = embeddings_for_query.astype('float32')

    if use_cosine_similarity:
        faiss.normalize_L2(xq)

    start = time()
    D, I = index.search(xq, num_neighbors)
    print("Elapsed time: {:.2f} s".format(time() - start))

    return D, I
