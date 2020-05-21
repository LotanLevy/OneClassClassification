from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_INTERPOLATION = cv2.INTER_LINEAR
MODEL_IMAGE_SIZE = (224, 224)

def read_labels_file(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        intlines = np.array([int(i) for i in lines])
    return intlines

def get_images_path(imdir):
    images = [join(imdir, f) for f in listdir(imdir) if (isfile(join(imdir, f)))]
    return images


def read_image(image_file, resize_image=()):
    cv_image = cv2.imread(image_file)
    if cv_image is None:
        raise RuntimeError(f"Unable to open {image_file}")

    if len(resize_image) > 0:
        cv_image = cv2.resize(cv_image, resize_image, interpolation=IMAGE_INTERPOLATION)
    cv_image = np.array(cv_image).astype("float32")
    cv_image /= 255.0

    return cv_image


class Dataloader:

    def __init__(self, image_paths_list, class_num, labels=None):
        if labels is None:
            labels = np.zeros((len(image_paths_list),))
        assert (len(image_paths_list) == len(labels))
        self._classes_num = class_num
        self._images_paths = image_paths_list
        self._labels = labels

    def read_images_batch(self, batch_size):
        rand_idx = np.random.randint(low=0, high=len(self._images_paths)-1, size=batch_size)
        batch_images = []
        for idx in rand_idx:
            batch_images.append(read_image(self._images_paths[idx], MODEL_IMAGE_SIZE))

        batch_labels = self._labels[rand_idx]
        hot_vecs = tf.keras.utils.to_categorical(batch_labels, num_classes=self._classes_num)
        return np.array(batch_images), hot_vecs


