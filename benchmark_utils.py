import time

from app_id_utils import freeze_app_ids, load_frozen_app_ids
from data_utils import get_data_path


def get_top_100_app_ids_file_name():
    # Get current day as yyyymmdd format
    date_format = "%Y%m%d"
    current_date = time.strftime(date_format)

    top_100_app_ids_file_name = get_data_path() + "{}_top_100_app_ids.txt".format(
        current_date
    )

    return top_100_app_ids_file_name


def save_top_100_app_ids(top_100_app_ids):
    output_file_name = get_top_100_app_ids_file_name()

    freeze_app_ids(app_ids=top_100_app_ids, output_file_name=output_file_name)

    return


def load_top_100_app_ids_from_disk():
    input_file_name = get_top_100_app_ids_file_name()

    top_100_app_ids = load_frozen_app_ids(input_file_name=input_file_name)

    return top_100_app_ids
