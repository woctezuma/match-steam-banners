from data_utils import load_sim_dict_from_disk
from match_utils import match_all
from print_utils import print_ranking_for_app_id, get_html_linked_image
from steam_spy_utils import load_game_names_from_steamspy, get_app_name


def load_sim_dict(use_cosine_similarity=True, pooling="avg"):
    try:
        # Caveat: the argument `use_cosine_similarity` was not tracked when sim_dict was saved to disk!
        # If you are unsure, run match_all() to ensure that `use_cosine_similarity` is set to the desired value.
        sim_dict = load_sim_dict_from_disk(pooling=pooling)
    except FileNotFoundError:
        sim_dict = match_all(
            use_cosine_similarity=use_cosine_similarity, pooling=pooling
        )

    return sim_dict


def extract_fixed_number_of_unique_games(sim_dict, num_outputs=250):
    sorted_app_ids = sorted(sim_dict.keys(), key=lambda x: sim_dict[x]["similarity"])

    # User-chosen num_outputs (N=250) allows to automatically set the similarity threshold so that N games are output.
    unique_app_ids = sorted_app_ids[:num_outputs]

    last_query_app_id = unique_app_ids[-1]
    similarity_threshold = sim_dict[last_query_app_id]["similarity"]

    print("Similarity threshold: {:.2f}".format(similarity_threshold))

    return unique_app_ids


def print_unique_games(sim_dict, unique_app_ids, game_names=None):
    if game_names is None:
        game_names = load_game_names_from_steamspy()

    num_elements_displayed = 1

    for query_app_id in unique_app_ids:
        matched_app_id = sim_dict[query_app_id]["app_id"]
        matched_app_id_as_list = [matched_app_id]

        print_ranking_for_app_id(
            query_app_id,
            matched_app_id_as_list,
            game_names=game_names,
            num_elements_displayed=num_elements_displayed,
        )

    return


def print_grid_of_unique_games(unique_app_ids, game_names=None):
    if game_names is None:
        game_names = load_game_names_from_steamspy()

    print("\n\nGrid\n\n")

    for query_app_id in unique_app_ids:
        query_app_name = get_app_name(query_app_id, game_names=game_names)
        html_linked_image = get_html_linked_image(query_app_id, query_app_name)

        print(html_linked_image)

    return


def find_unique_games(use_cosine_similarity=True, pooling="avg", num_outputs=250):
    sim_dict = load_sim_dict(
        use_cosine_similarity=use_cosine_similarity, pooling=pooling
    )

    unique_app_ids = extract_fixed_number_of_unique_games(
        sim_dict, num_outputs=num_outputs
    )

    print_unique_games(sim_dict, unique_app_ids)

    print_grid_of_unique_games(unique_app_ids)

    return


if __name__ == "__main__":
    find_unique_games(num_outputs=250)
