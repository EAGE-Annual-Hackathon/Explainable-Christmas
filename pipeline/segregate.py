import numpy as np
from sklearn import preprocessing, model_selection


def main():
    """ Segregate and transform the data into training and testing sets."""
    
    try:
        x_data = np.load("pipeline/outputs/attributes_x_16x16.npy")
        y_data = np.load("pipeline/outputs/y_16x16.npy")
    except:
        x_data = np.load("pipeline/outputs/attributes_x_64x64.npy")
        y_data = np.load("pipeline/outputs/y_64x64.npy")
        x_data = x_data[:, ::4, ::4, :]
        y_data = y_data[:, ::4, ::4]


    # to recreate the original shape we'll need to store it
    n_samples, img_x, img_y, n_channels = x_data.shape

    # train-test split
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x_data, y_data, test_size=0.2
        )

    # Fit transformer to the training set then apply it to test set
    transformer = preprocessing.PowerTransformer()
    x_train = transformer.fit_transform(x_train.reshape(-1, n_channels))
    x_train = x_train.reshape(-1, img_x, img_y, n_channels)

    x_test = transformer.transform(x_test.reshape(-1, n_channels))
    x_test = x_test.reshape(-1, img_x, img_y, n_channels)

    np.save("pipeline/outputs/x_train_transformed.npy", x_train)
    np.save("pipeline/outputs/y_train_transformed.npy", y_train)
    np.save("pipeline/outputs/x_test_transformed.npy", x_test)
    np.save("pipeline/outputs/y_test_transformed.npy", y_test)

if __name__ == "__main__":

    main()