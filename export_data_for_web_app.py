import json
from pathlib import Path

import numpy as np

from app_id_utils import get_frozen_app_ids
from data_utils import get_label_database_filename
from faiss_utils import get_faiss_search_structure, find_faiss_knn_for_all


def get_export_folder_name():
    export_folder_name = "data_export/"
    Path(export_folder_name).mkdir(exist_ok=True)

    return export_folder_name


def export_app_ids(out_fname_json="app_ids.json", out_fname_npy="app_ids.npy"):
    app_ids = [int(app_id) for app_id in get_frozen_app_ids()]

    print("#apps = {}".format(len(app_ids)))

    with open(get_export_folder_name() + out_fname_json, "w", encoding="utf8") as f:
        json.dump(app_ids, f)

    # As of January 2021, the largest appID on Steam is less than 2 millions, so uint32 is ok (max value: ~ 4.2 billion)
    v = np.asarray(app_ids, dtype=np.uint32)

    np.save(
        get_export_folder_name() + out_fname_npy,
        v,
        allow_pickle=False,
        fix_imports=False,
    )

    return app_ids


def export_app_names(
    app_ids, input_fname="IStoreService.json", out_fname="app_names.json"
):
    with open(input_fname, "r", encoding="utf8") as f:
        data = json.load(f)

    app_info = dict()
    for e in data["response"]["apps"]:
        app_info[e["appid"]] = e["name"]

    app_names = []
    for app_id in app_ids:
        try:
            app_name = app_info[int(app_id)]
        except KeyError:
            # This can happen if I reuse an old folder of images without reusing the JSON files corresponding to it.
            # For instance, Grand Theft Auto: Vice City (appID=12240) was removed from Steam on April 19, 2021.
            # So, if I re-use pictures downloaded in January 2021, and add new pictures downloaded in May 2021,
            # but I do not re-use JSON from January, and only rely on JSON from May, then that app name will be unknown.
            app_name = "N/A"
        app_names.append(app_name)

    print("#apps = {}".format(len(app_names)))

    with open(get_export_folder_name() + out_fname, "w") as f:
        json.dump(app_names, f)

    return app_names


def export_matches(
    num_neighbors=100, use_cosine_similarity=True, out_fname="matches_faiss.npy"
):
    embeddings = np.load(get_label_database_filename())

    print("(#apps, #features) = {}".format(embeddings.shape))

    index = get_faiss_search_structure(
        embeddings, use_cosine_similarity=use_cosine_similarity
    )

    D, I = find_faiss_knn_for_all(
        index, embeddings, num_neighbors, use_cosine_similarity=use_cosine_similarity
    )

    # As of January 2021, the largest list index is about ~ 30 k (#apps), so uint16 is ok (max value: ~ 65 k)
    v = I.astype("uint16")

    np.save(
        get_export_folder_name() + out_fname, v, allow_pickle=False, fix_imports=False
    )

    return


if __name__ == "__main__":
    app_ids = export_app_ids()

    input_fname = "IStoreService.json"
    if Path(input_fname).exists():
        app_names = export_app_names(app_ids, input_fname=input_fname)

    export_matches(num_neighbors=100, use_cosine_similarity=True)
