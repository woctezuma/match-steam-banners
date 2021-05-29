import argparse

import torch
import vision_transformer as vits
from PIL import Image
from torchvision import transforms as pth_transforms
from utils import bool_flag


def get_default_feature_style():
    # Either 'simple' or 'complex'.
    return "simple"


def get_default_arch():
    # Either 'vit_small' or 'vit_base'.
    return "vit_small"


def get_default_patch_size():
    # Either 16 or 8.
    return 16


def get_default_n_last_blocks(arch):
    if "small" in arch:
        n_last_blocks = 4
    else:
        n_last_blocks = 1
    return n_last_blocks


def get_default_avgpool_patchtokens(arch):
    return bool("base" in arch)


def get_parser_args(
    feature_style=None,
    arch=None,
    patch_size=None,
    n_last_blocks=None,
    avgpool_patchtokens=None,
):
    if feature_style is None:
        feature_style = get_default_feature_style()

    if arch is None:
        arch = get_default_arch()

    if patch_size is None:
        patch_size = get_default_patch_size()

    if n_last_blocks is None:
        n_last_blocks = get_default_n_last_blocks(arch)

    if avgpool_patchtokens is None:
        avgpool_patchtokens = get_default_avgpool_patchtokens(arch)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base"],
    )
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--n_last_blocks", default=4, type=int)
    parser.add_argument("--avgpool_patchtokens", default=False, type=bool_flag)
    parser.add_argument(
        "--feature_style",
        default="simple",
        type=str,
        choices=["simple", "complex", "vit_base"],
    )
    args = parser.parse_args(
        [
            "--arch",
            arch,
            "--patch_size",
            str(patch_size),
            "--n_last_blocks",
            str(n_last_blocks),
            "--avgpool_patchtokens",
            str(avgpool_patchtokens),
            "--feature_style",
            str(feature_style),
        ]
    )

    return args


def get_dino_url(args=None):
    if args is None:
        args = get_parser_args()

    if args.arch == "vit_small" and args.patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif args.arch == "vit_small" and args.patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
    elif args.arch == "vit_base" and args.patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif args.arch == "vit_base" and args.patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    else:
        url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"

    return url


def get_model_slug_for_dino():
    return "dino"


def count_num_features_for_dino(model=None, args=None):
    if args is None:
        args = get_parser_args()

    if model is None:
        model = get_dino_model_name(args)

    if args.feature_style == "simple":
        num_features = model.embed_dim
    else:
        # Reference: https://github.com/facebookresearch/dino/blob/main/eval_linear.py
        num_features = model.embed_dim * (
            args.n_last_blocks + int(args.avgpool_patchtokens)
        )

    return num_features


def get_model_resolution_for_dino(model=None):
    resolution = 224
    return resolution


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def get_dino_model_name(args=None):
    if args is None:
        args = get_parser_args()

    model_name = "ViT-{}/{}".format(args.arch[0].upper(), args.patch_size)

    return model_name


def load_dino_tools(args=None):
    if args is None:
        args = get_parser_args()

    model = get_model_for_dino(args)
    preprocess = get_preprocessing_for_dino()

    return model, preprocess


def get_preprocessing_for_dino():
    # Reference: https://github.com/facebookresearch/dino/blob/main/eval_linear.py

    preprocess = pth_transforms.Compose(
        [
            pth_transforms.Resize(
                256, interpolation=pth_transforms.InterpolationMode.BICUBIC
            ),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return preprocess


def preprocess_image_array_for_model_for_dino(image_array, preprocess=None):
    if preprocess is None:
        preprocess = get_preprocessing_for_dino()

    pil_img = Image.fromarray(image_array)
    inp = preprocess(pil_img)
    processed_array = torch.unsqueeze(inp, axis=0)

    return processed_array


def get_model_for_dino(input_shape=None, include_top=None, pooling=None, args=None):
    if args is None:
        args = get_parser_args()

    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(get_device())

    url = get_dino_url(args)
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/dino/" + url
    )
    model.load_state_dict(state_dict, strict=True)

    return model


def label_image_for_dino(
    image, model=None, preprocess=None, normalize_features=True, args=None
):
    if args is None:
        args = get_parser_args()

    if model is None:
        model = get_model_for_dino(args=args)

    preprocessed_image = preprocess_image_array_for_model_for_dino(
        image, preprocess=preprocess
    )
    image_array = preprocessed_image.to(get_device(), non_blocking=True)

    # forward
    if args.feature_style == "simple":
        # Reference: https://github.com/facebookresearch/dino/blob/main/eval_knn.py
        with torch.no_grad():
            image_features = model(image_array)
    else:
        # Reference: https://github.com/facebookresearch/dino/blob/main/eval_linear.py
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(
                image_array, args.n_last_blocks
            )
            output = [x[:, 0] for x in intermediate_output]
            if args.avgpool_patchtokens:
                output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
            image_features = torch.cat(output, dim=-1)

    if normalize_features:
        image_features /= image_features.norm(dim=-1, keepdim=True)

    yhat = image_features.cpu().numpy()

    return yhat


if __name__ == "__main__":
    slug_name = get_model_slug_for_dino()
    print("Slug: {}".format(slug_name))
