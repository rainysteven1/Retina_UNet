import numpy as np
from PIL import Image


def visuliaze_sample_input(data, file_name):
    if data.shape[-1] == 1:
        data = np.reshape(data, data.shape[:-1])
    data = (data if np.max(data) > 1 else data * 255).astype(np.uint8)
    img = Image.fromarray(data)
    img.save(file_name)
