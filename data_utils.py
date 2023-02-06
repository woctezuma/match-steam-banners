import json
from pathlib import Path


def get_data_path():
    data_path = "data/"
    Path(data_path).mkdir(exist_ok=True)

    return data_path


def get_image_data_path(is_horizontal_banner=False):
    keyword = "horizontal" if is_horizontal_banner else "vertical"

    image_data_path = get_data_path() + f"resized_{keyword}_steam_banners/"

    return image_data_path


def get_image_extension():
    image_extension = ".jpg"

    return image_extension


def get_label_database_filename(pooling="avg"):
    pooling_str = "" if pooling is None else "." + pooling

    label_database_filename = get_data_path() + "label_database{}.npy".format(
        pooling_str,
    )

    return label_database_filename


def get_unique_games_file_name(pooling="avg"):
    pooling_str = "" if pooling is None else "." + pooling

    unique_games_file_name = get_data_path() + f"unique_games{pooling_str}.json"

    return unique_games_file_name


def save_sim_dict(sim_dict, pooling="avg"):
    with open(get_unique_games_file_name(pooling=pooling), "w", encoding="utf8") as f:
        json.dump(sim_dict, f)

    return


def load_sim_dict_from_disk(pooling="avg"):
    with open(get_unique_games_file_name(pooling=pooling), encoding="utf8") as f:
        sim_dict = json.load(f)

    return sim_dict
