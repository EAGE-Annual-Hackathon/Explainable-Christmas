import numpy as np

from sklearn import preprocessing
from sklearn import model_selection

def main():
    """ Segregate and transform the data into training and testing sets."""
    # x_data = np.load("pipeline/data/x.npy")
    
    x_data = np.load("pipeline/data/attributes_array_0-100.npy")
    x_data = np.nan_to_num(x_data)
    print(np.isnan(x_data.sum()))
    y_data = np.load("pipeline/data/y.npy")
    y_data = y_data[:100]

    # reshape the data, consider that it has 4 dimensions
    #(sample,img,img, channels)
    n_samples, img_x, img_y, n_channels = x_data.shape
    print

    # train-test split
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x_data, y_data, test_size=0.2
        )

    transformer = preprocessing.PowerTransformer()

    x_train = transformer.fit_transform(x_data.reshape(-1, n_channels))
    x_train = x_train.reshape(n_samples, img_x, img_y, n_channels)

    x_test = transformer.transform(x_test.reshape(-1, n_channels))
    x_test = x_test.reshape(-1, img_x, img_y, n_channels)

    print(np.isnan(x_train.sum()))
    print(np.isnan(x_test.sum()))    

    np.save("pipeline/data/x_train_transformed.npy", x_train)
    np.save("pipeline/data/y_train_transformed.npy", y_train)
    np.save("pipeline/data/x_test_transformed.npy", x_test)
    np.save("pipeline/data/y_test_transformed.npy", y_test)

if __name__ == "__main__":

    main()