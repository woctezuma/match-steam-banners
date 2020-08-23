def get_size_to_name_conversion_table():
    size_to_name_conversion_table = {
        # horizontal
        "460x215": "header",  # ratio 2.14
        "231x87": "capsule_231x87",  # ratio 2.66
        "467x181": "capsule_467x181",  # ratio 2.58
        "616x353": "capsule_616x353",  # ratio 1.75
        # vertical
        "300x450": "library_600x900",  # ratio 0.67
        "600x900": "library_600x900_2x",  # ratio 0.67
    }

    return size_to_name_conversion_table


def get_banner_conventional_name(is_horizontal_banner=False):
    if is_horizontal_banner:
        banner_size = "616x353"
    else:
        banner_size = "300x450"

    conversion_table = get_size_to_name_conversion_table()

    banner_conventional_name = conversion_table[banner_size]

    return banner_conventional_name


def get_banner_url(app_id, is_horizontal_banner=False):
    # Caveat: vertical banners are a recent addition to Steam, and they do not exist for every game!

    banner_url = "https://steamcdn-a.akamaihd.net/steam/apps/{}/{}.jpg".format(
        app_id, get_banner_conventional_name(is_horizontal_banner)
    )

    return banner_url


def get_store_url(app_id):
    store_url = "https://store.steampowered.com/app/{}".format(app_id)

    return store_url
