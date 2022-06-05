import numpy as np

from sklearn import preprocessing

def main():
    """ Segregate and transform the data into training and testing sets."""
    x_data = np.load("pipeline/data/x.npy")
    y_data = np.load("pipeline/data/y.npy")

    transformer = preprocessing.PowerTransformer()

    # reshape the data, consider that it has 4 dimensions
    transformer.fit(x_data.)

    x_data = preprocessing.scale(x_data)
    