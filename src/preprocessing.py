import cv2
import numpy as np

CONFIGURATIONS = {"length": 4, "channels": 3, "gray_channel": 1}


def preprocess(data: np.ndarray):
    imgs = rgb2gray(data)
    imgs = dataset_normalized(imgs)
    imgs = clahe_equalized(imgs)
    imgs = adjust_gamma(imgs, 1.2)
    return imgs / 255.0


def rgb2gray(data):
    bn_imgs = (
        data[:, 0, :, :] * 0.299 + data[:, 1, :, :] * 0.587 + data[:, 2, :, :] * 0.114
    )
    bn_imgs = np.reshape(bn_imgs, [data.shape[0], 1, data.shape[2], data.shape[3]])
    return bn_imgs


# normalize over the dataset
def dataset_normalized(data):
    data_std, data_mean = np.std(data), np.mean(data)
    data_normalized = (data - data_mean) / data_std
    for i in range(data.shape[0]):
        data_normalized[i] = (
            (data_normalized[i] - np.min(data_normalized[i]))
            / (np.max(data_normalized[i]) - np.min(data_normalized[i]))
        ) * 255.0
    return data_normalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe_equalized(data):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    data_equalized = np.empty(data.shape)
    for i in range(data.shape[0]):
        data_equalized[i, 0] = clahe.apply(np.array(data[i, 0], dtype=np.uint8))
    return data_equalized


def adjust_gamma(data, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    new_data = np.empty(data.shape)
    for i in range(data.shape[0]):
        new_data[i, 0] = cv2.LUT(np.array(data[i, 0], dtype=np.uint8), table)
    return new_data
