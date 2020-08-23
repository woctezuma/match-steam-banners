import os
from pathlib import Path

from data_utils import get_data_path, get_image_data_path, get_image_extension


def app_id_to_image_filename(app_id, is_horizontal_banner=False):
    image_data_path = get_image_data_path(is_horizontal_banner)

    image_filename = image_data_path + str(app_id) + get_image_extension()

    return image_filename


def image_filename_to_app_id(image_filename):
    base_name = os.path.basename(image_filename)

    app_id = base_name.strip(get_image_extension())

    return app_id


def list_app_ids(is_horizontal_banner=False):
    image_data_path = get_image_data_path(is_horizontal_banner)

    image_filenames = Path(image_data_path).glob("*" + get_image_extension())

    app_ids = [image_filename_to_app_id(filename) for filename in image_filenames]

    app_ids.sort(key=int)

    return app_ids


def get_frozen_app_ids_filename():
    frozen_app_ids_filename = get_data_path() + "frozen_app_ids.txt"

    return frozen_app_ids_filename


def freeze_app_ids(app_ids, output_file_name=None):
    if output_file_name is None:
        output_file_name = get_frozen_app_ids_filename()

    with open(output_file_name, "w", encoding="utf8") as f:
        for app_id in app_ids:
            f.write("{}\n".format(app_id))

    return


def load_frozen_app_ids(input_file_name=None):
    if input_file_name is None:
        input_file_name = get_frozen_app_ids_filename()

    with open(input_file_name, "r", encoding="utf8") as f:
        # Do not convert to a set object, or any other conversion, because we want to keep the list order as it is.
        # Just read the list from the file. That is all there is to do. Otherwise, appIDs will be scrambled!
        frozen_app_ids = [app_id.strip() for app_id in f.readlines()]

    return frozen_app_ids


def get_frozen_app_ids(is_horizontal_banner=False):
    try:
        frozen_app_ids = load_frozen_app_ids()
    except FileNotFoundError:
        print("Creating {}".format(get_frozen_app_ids_filename()))
        frozen_app_ids = list_app_ids(is_horizontal_banner=is_horizontal_banner)
        freeze_app_ids(frozen_app_ids)

    return frozen_app_ids
