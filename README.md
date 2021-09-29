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
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
NB: Tensorflow 2 supports Python 3.5–3.8, not 3.9.

-   Install OpenAI's CLIP with pip:
```bash
pip install git+https://github.com/openai/CLIP.git
```

-   Install Facebook's DINO with git:

```bash
git clone https://github.com/facebookresearch/dino.git
mv dino/vision_transformer.py .
mv dino/utils.py .
```

## Model

MobileNet is a convolutional neural network, trained on ImageNet-1k (1.28M images with 1000 classes).

In this repository, the image encoder is `MobileNetV3-Small`.

## Data

Data is available in [`download-steam-banners-data/`](https://github.com/woctezuma/download-steam-banners-data).

The most recent data snapshot was downloaded in August 2020 with [this Colab notebook][download_steam_banners].
[![Open In Colab][colab-badge]][download_steam_banners]
It consists of 19,049 **vertical** Steam banners resized from 300x450 to 256x256 resolution.

## Usage

NB: a [Colab notebook][match_steam_banners-notebook] is available in my [`colab`][colab-branch] branch.
[![Open In Colab][colab-badge]][match_steam_banners-notebook]

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

### 4. Export data and results for a web app

Optionally, export data and results for [a web app][my-flask-API]:

```bash
python export_data_for_web_app.py
```

## Results

Results obtained with [MobileNet v3][keras-mobilenet] are shown [on the Wiki][my-wiki].

The linked pages contain a lot of images and might be slow to load depending on your Internet bandwidth.

### Similar games

Direct links to similarity results are available below:
-   for each game, find [the 10 most similar games](https://github.com/woctezuma/match-steam-banners/wiki/Benchmark-top100).

For instance:
![Similar vertical (Call of Duty)](https://github.com/woctezuma/match-steam-banners/wiki/img/similar_cod.jpg)

Or:
![Similar vertical banners (Half-Life)](https://github.com/woctezuma/match-steam-banners/wiki/img/similar_hl.jpg)

### Unique games

Direct links to similarity results are available below:
-   for each unique game, display [the 1 most similar game](https://github.com/woctezuma/match-steam-banners/wiki/Unique-Games),
-   a [grid of unique games](https://github.com/woctezuma/match-steam-banners/wiki/Grid-of-Unique-Games).

Unique games seem to have in common that their banner contains mostly a texture and a title, without any large character, item or shape.

For instance:
![Unique vertical banners](https://github.com/woctezuma/match-steam-banners/wiki/img/unique_games.jpg)

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
[my-flask-API]: <https://github.com/woctezuma/heroku-flask-api>

[colab-branch]: <https://github.com/woctezuma/match-steam-banners/tree/colab>
[match_steam_banners-notebook]: <https://colab.research.google.com/github/woctezuma/match-steam-banners/blob/colab/notebooks/match_steam_banners.ipynb>

[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
