# Objective: download query images which are missing from our data snapshot.
# This is necessary if:
# -   you query a **recent** game, typically found in SteamSpy's top 100 most played games during the past two weeks,
# -   the data snapshot was downloaded when i) the game was not yet on the store, or ii) did not have a vertical banner.
#
# NB: To download many images, you should rather use asynchronous libraries instead of `requests`, as shown in:
#     https://colab.research.google.com/github/woctezuma/google-colab/blob/master/download_steam_banners.ipynb

import requests

from app_id_utils import app_id_to_image_filename
from steam_store_utils import get_banner_url


def download_query_image(app_id, is_horizontal_banner=False, output_filename=None):
    # Reference: https://stackoverflow.com/a/21595698

    if output_filename is None:
        output_filename = app_id_to_image_filename(app_id, is_horizontal_banner)

    banner_url = get_banner_url(app_id, is_horizontal_banner=is_horizontal_banner)

    print("Downloading banner for appID={} from {}".format(app_id, banner_url))

    response_data = requests.get(url=banner_url)

    status_code = response_data.status_code

    if status_code == 200:
        image_content = response_data.content

        print("Saving image content to {}".format(output_filename))

        with open(output_filename, "wb") as f:
            f.write(image_content)

    else:
        print("Download failed with status code {}.".format(status_code))

    return


if __name__ == "__main__":
    app_id = 1250410  # Microsoft Flight Simulator
    download_query_image(
        app_id, output_filename="temp_vertical_banner.jpg", is_horizontal_banner=False
    )
    download_query_image(
        app_id, output_filename="temp_horizontal_banner.jpg", is_horizontal_banner=True
    )
