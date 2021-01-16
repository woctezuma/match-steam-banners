from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def load_image(image_filename, target_size):
    image = load_img(image_filename, target_size=target_size)
    return image


def convert_image_to_array(image):
    image_array = img_to_array(image)
    return image_array


def preprocess_image_array_for_model(image_array):
    image_array = preprocess_input(image_array)
    return image_array


def get_model(input_shape, include_top, pooling):
    model = MobileNetV3Small(input_shape=input_shape, include_top=include_top, pooling=pooling)
    return model
