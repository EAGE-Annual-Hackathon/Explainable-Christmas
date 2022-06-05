import time
from pathlib import Path

import bruges as bg
import numpy as np
from scipy.ndimage import generic_filter, sobel
from skimage import io


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

def main():
    """ Create numpy arrays from the train images and labels."""
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
    
    np.save("pipeline/data/x.npy", x_data)
    np.save("pipeline/data/y.npy", y_data)

    n_samples, img_x, img_y = x_data.shape[0], 101, 101
    n_channels = 4
    
    # create attributes
    now = time.time()
    attributes = np.zeros(shape=(n_samples, img_x, img_y, n_channels), dtype=np.float32)
    for index in range(x_data.shape[0]):
        attributes[index] = run_attributes(x_data[index])

        if index%5 == 0:
           print(f'{index}th Image Complete in {(time.time() - now)/60} Minutes')
        
    np.save("pipeline/data/attributes.npy")

if __name__ == "__main__":
    main()


