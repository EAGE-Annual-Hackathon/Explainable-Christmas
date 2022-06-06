import time
from pathlib import Path

import bruges as bg
import numpy as np
from scipy.ndimage import generic_filter, sobel
from skimage import io
from skimage.transform import resize


def rms(data):
    data = np.asanyarray(data)
    return np.sqrt(np.sum(data**2) / data.size)

def attribute_wrapper(input_data, attribute_func, output_shape=None, **kwargs):
    image = np.reshape(input_data, newshape=(1, 101, 101))
    attribute = attribute_func(image, **kwargs)
    
    if output_shape is None:
        return np.reshape(attribute, input_data.shape)
    else:
        return np.reshape(attribute, output_shape)
    
def run_attributes(input_data):
    energy = attribute_wrapper(input_data, bg.attribute.energy, duration=5, dt=1)
    semb = attribute_wrapper(input_data, bg.attribute.similarity, duration=5, dt=1, step_out=3, kind='gst')
    sobel_image = sobel(input_data)

    return np.stack([input_data, energy, semb, sobel_image], axis=-1)

def clean_and_resize_data(attributes, y):
    attributes_x_reshaped = np.zeros(shape=(3747, 64, 64, 4), dtype=np.float32)
    attributes_y_reshaped = np.zeros(shape=(3747, 64, 64, 1), dtype=np.float32)

    j=0
    k=0
    for i in range(attributes.shape[0]):
        if (np.isnan(attributes[i]).any()):
            j+=1
        else:
            attributes_x_reshaped[k] = resize(attributes[i], (64, 64, 4), anti_aliasing=True)
            attributes_y_reshaped[k] = resize(y[i], (64, 64, 1), anti_aliasing=True)
            k+=1
            
        if i % 100 == 0:
            print(i)
    
    return attributes_x_reshaped, attributes_y_reshaped
