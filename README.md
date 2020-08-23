# Match Steam Banners

[![Build status with Github Action][build-image-action]][build-action]
[![Updates][dependency-image]][pyup]
[![Python 3][python3-image]][pyup]
[![Code coverage][codecov-image]][codecov]
[![Code Quality][codacy-image]][codacy]

This repository contains Python code to retrieve Steam games with similar store banners.

It is based on the best practices observed in my previous repository [`download-steam-banners`][my-previous-repository]:
-   features extracted by a neural network (MobileNet v1 previously ; v3 here),
-   global average pooling of features, as in MobileNet papers and [implementations][keras-mobilenet],
-   cosine similarity.

![Similar vertical banners](https://github.com/woctezuma/match-steam-banners/wiki/img/illustration.jpg)

## Requirements

-   Install the latest version of [Python 3.X](https://www.python.org/downloads/).
-   Install the required packages:

```bash
python -m pip install --upgrade pip
pip install --upgrade cython
pip install -r requirements.txt
```

## Data

Data is available in [`download-steam-banners-data/`](https://github.com/woctezuma/download-steam-banners-data).

The most recent data snapshot was downloaded in August 2020 with [this Colab notebook][download_steam_banners].
It consists of 19,049 **vertical** Steam banners resized from 300x450 to 256x256 resolution.

## Usage

### 1. Features

First, compute and store the 1024 features corresponding to each banner:

```bash
python build_feature_index.py
```

### 2. Similar games

Find the **10** most similar store banners to curated query appIDs:

```bash
python retrieve_similar_features.py
```

NB: by default, query appIDs consist of:
-   the top 100 most played games during the past two weeks, according to [SteamSpy][steamspy-api],
-   a few manually curated games.

### 3. Unique games

Find the **one** most similar store banner to all appIDs available on the store, then display the most unique games:

```bash
python find_unique_games.py
```

NB: *unique* games are ones which are the most dissimilar (low similarity score) to others to their first neighbor.

## Results

Results obtained with [MobileNet v3][keras-mobilenet] are shown [on the Wiki][my-wiki].

## References

-   [`download-steam-banners`][my-previous-repository]: retrieve Steam games with similar store banners,
-   [`download-steam-screenshots`][screenshot-repository]: retrieve Steam games with similar store screenshots,

<!-- Definitions -->

[build]: <https://travis-ci.org/woctezuma/match-steam-banners>
[build-image]: <https://travis-ci.org/woctezuma/match-steam-banners.svg?branch=master>

[build-action]: <https://github.com/woctezuma/match-steam-banners/actions>
[build-image-action]: <https://github.com/woctezuma/match-steam-banners/workflows/Python application/badge.svg?branch=master>

[pyup]: <https://pyup.io/repos/github/woctezuma/match-steam-banners/>
[dependency-image]: <https://pyup.io/repos/github/woctezuma/match-steam-banners/shield.svg>
[python3-image]: <https://pyup.io/repos/github/woctezuma/match-steam-banners/python-3-shield.svg>

[codecov]: <https://codecov.io/gh/woctezuma/match-steam-banners>
[codecov-image]: <https://codecov.io/gh/woctezuma/match-steam-banners/branch/master/graph/badge.svg>

[codacy]: <https://www.codacy.com/app/woctezuma/match-steam-banners>
[codacy-image]: <https://api.codacy.com/project/badge/Grade/66348d16574146a298ec81ec2d626efe>

[my-previous-repository]: <https://github.com/woctezuma/download-steam-banners>
[keras-mobilenet]: <https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v3.py>
[download_steam_banners]: <https://colab.research.google.com/github/woctezuma/google-colab/blob/master/download_steam_banners.ipynb>
[my-wiki]: <https://github.com/woctezuma/match-steam-banners/wiki>
[screenshot-repository]: <https://github.com/woctezuma/download-steam-screenshots>
[steamspy-api]: <https://github.com/woctezuma/steamspypi>
