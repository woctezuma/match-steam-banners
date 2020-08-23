import steamspypi

from benchmark_utils import save_top_100_app_ids, load_top_100_app_ids_from_disk


def load_game_names_from_steamspy():
    data = steamspypi.load()

    game_names = dict()
    for app_id in data.keys():
        game_names[app_id] = data[app_id]["name"]

    return game_names


def get_app_name(app_id, game_names=None):
    if game_names is None:
        game_names = load_game_names_from_steamspy()

    try:
        app_name = game_names[str(app_id)]
    except KeyError:
        app_name = "Unknown"

    return app_name


def get_top_100_app_ids():
    data_request = dict()
    data_request["request"] = "top100in2weeks"

    data = steamspypi.download(data_request)

    top_100_app_ids = list(data.keys())

    return top_100_app_ids


def load_top_100_app_ids():
    try:
        top_100_app_ids = load_top_100_app_ids_from_disk()
    except FileNotFoundError:
        top_100_app_ids = get_top_100_app_ids()
        save_top_100_app_ids(top_100_app_ids)

    return top_100_app_ids


def load_benchmarked_app_ids(append_hard_coded_app_ids=True):
    # Reference: https://github.com/woctezuma/steam-descriptions/blob/master/benchmark_utils.py

    top_100_app_ids = load_top_100_app_ids()

    # Append hard-coded appIDs

    additional_app_ids = [
        "620",
        "364470",
        "504230",
        "583950",
        "646570",
        "863550",
        "794600",
        "814380",
    ]

    benchmarked_app_ids = sorted(top_100_app_ids)
    if append_hard_coded_app_ids:
        for app_id in set(additional_app_ids).difference(top_100_app_ids):
            benchmarked_app_ids.append(app_id)

    return benchmarked_app_ids


if __name__ == "__main__":
    game_names = load_game_names_from_steamspy()
    top_100_app_ids = load_top_100_app_ids()
