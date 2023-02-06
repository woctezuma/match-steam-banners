import clip
import torch


def get_model_slug_for_clip():
    return "clip"


def count_num_features_for_clip(model=None):
    num_features = 512
    return num_features


def get_model_resolution_for_clip(model=None):
    if model is None:
        resolution = 224
    else:
        resolution = model.input_resolution.item()
    return resolution


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def get_clip_model_name():
    model_name = "ViT-B/32"
    return model_name


def load_clip_tools():
    model, preprocess = clip.load(get_clip_model_name(), device=get_device())
    return model, preprocess


def get_preprocessing_for_clip():
    _, preprocess = load_clip_tools()
    return preprocess


def preprocess_image_array_for_model_for_clip(image_array, preprocess=None):
    if preprocess is None:
        preprocess = get_preprocessing_for_clip()

    image_array = preprocess(image_array).unsqueeze(0)
    return image_array


def get_model_for_clip(input_shape=None, include_top=None, pooling=None):
    model, _ = load_clip_tools()
    return model


def label_image_for_clip(image, model=None, preprocess=None, normalize_features=True):
    if model is None:
        model = get_model_for_clip()

    preprocessed_image = preprocess_image_array_for_model_for_clip(
        image,
        preprocess=preprocess,
    )
    image_array = preprocessed_image.to(get_device())

    with torch.no_grad():
        image_features = model.encode_image(image_array)

    if normalize_features:
        image_features /= image_features.norm(dim=-1, keepdim=True)

    yhat = image_features.cpu().numpy()

    return yhat


if __name__ == "__main__":
    slug_name = get_model_slug_for_clip()
    print("Slug: {}".format(slug_name))
