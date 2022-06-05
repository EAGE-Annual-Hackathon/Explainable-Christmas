from pathlib import Path

import numpy as np
from skimage import io


def main():
    """ Create numpy arrays from the train images and labels."""
    train_images = []
    train_labels = []

    train_path = Path('pipeline/data/train/images')
    labels_path = Path('pipeline/data/train/labels')

    for image in train_path.iterdir():
        _ = io.imread(image)
        train_images.append(_)
        _ = io.imread(labels_path / image.name)
        train_labels.append(_)

    x_data = np.array(train_images)
    y_data = np.array(train_labels)
    
    np.save("pipeline/data/x.npy", x_data)
    np.save("pipeline/data/y.npy", y_data)

if __name__ == "__main__":
    main()
