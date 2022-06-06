import time
from pathlib import Path

import bruges as bg
import numpy as np
from scipy.ndimage import generic_filter, sobel
from skimage import io
from skimage.transform import resize

from utils import rms, attribute_wrapper, run_attributes, clean_and_resize_data


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


def main():
    """ Create numpy arrays from the train images and labels and create attributes."""
    train_images = []
    train_labels = []

    # print(Path.cwd())
    train_path = Path('pipeline/data/train/images')
    labels_path = Path('pipeline/data/train/masks')

    for image in train_path.iterdir():
        train_data = io.imread(image, as_gray=True)
        train_images.append(train_data)
        label_data = io.imread(labels_path / image.name, as_gray=True)
        train_labels.append(label_data)

    x_data = np.array(train_images)
    y_data = np.array(train_labels)
    
    np.save("pipeline/outputs/x.npy", x_data)
    np.save("pipeline/outputs/y.npy", y_data)

    n_samples, img_x, img_y = x_data.shape[0], 101, 101
    n_channels = 4
    
    # create attributes
    now = time.time()
    attributes = np.zeros(shape=(n_samples, img_x, img_y, n_channels), dtype=np.float32)
    for index in range(x_data.shape[0]):
        attributes[index] = run_attributes(x_data[index])

        if index%5 == 0:
           print(f'{index}th Image Complete in {(time.time() - now)/60} Minutes')
        
    np.save("pipeline/outputs/attributes.npy", attributes)

    attributes_x_reshaped, y_resized = clean_and_resize_data(attributes, y_data)
    np.save('pipeline/outputs/attributes_x_64x64.npy', attributes_x_reshaped)
    np.save('pipeline/outputs/y_64x64.npy', y_resized)

    # Resizing it once again from 64x64 to 16x16 due to computational limits
    attributes_x_reshaped = attributes_x_reshaped[:, ::4, ::4, :]
    y_resized = y_resized[:, ::4, ::4]

    np.save('pipeline/outputs/attributes_x_16x16.npy', attributes_x_reshaped)
    np.save('pipeline/outputs/y_16x16.npy', y_resized)

if __name__ == "__main__":

    main()


