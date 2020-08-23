from steam_spy_utils import load_game_names_from_steamspy, get_app_name
from steam_store_utils import get_store_url, get_banner_url


def get_default_image_width():
    default_image_width = 150

    return default_image_width


def get_bb_code_linked_image(app_id):
    image_link_str = '[url={}][img="width:{}px;"]{}[/img][/url]'

    bb_code_linked_image = image_link_str.format(
        get_store_url(app_id), get_default_image_width(), get_banner_url(app_id)
    )

    return bb_code_linked_image


def get_html_linked_image(app_id, app_name=None):
    if app_name is None:
        app_name = "Unknown"

    # Reference: https://stackoverflow.com/a/14747656
    image_link_str = '[<img alt="{}" src="{}" width="{}">]({})'

    html_linked_image = image_link_str.format(
        app_name,
        get_banner_url(app_id),
        get_default_image_width(),
        get_store_url(app_id),
    )

    return html_linked_image


def print_ranking_for_app_id(
    query_app_id, reference_app_id_counter, game_names=None, num_elements_displayed=10,
):
    if game_names is None:
        game_names = load_game_names_from_steamspy()

    query_app_name = get_app_name(query_app_id, game_names=game_names)
    html_linked_image = get_html_linked_image(query_app_id, query_app_name)

    print("\nQuery:\n\n{}\n\n".format(html_linked_image))

    for rank, app_id in enumerate(reference_app_id_counter, start=1):
        app_name = get_app_name(app_id, game_names=game_names)
        html_linked_image = get_html_linked_image(app_id, app_name)

        print(html_linked_image, end="")

        # Display results on two rows
        if rank == num_elements_displayed / 2:
            print("\n")

        if rank >= num_elements_displayed:
            print("\n")
            break

    return
