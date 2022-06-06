import mlflow

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from build_model import build_model, build_simpler_model


def main():

    mlflow.tensorflow.autolog()

    x_train = np.load("pipeline/outputs/x_train_transformed.npy")
    x_test = np.load("pipeline/outputs/x_test_transformed.npy")
    y_train = np.load("pipeline/outputs/y_train_transformed.npy")
    y_test = np.load("pipeline/outputs/y_test_transformed.npy")

    y_train = np.where(y_train == 0, 0, 1)
    y_test = np.where(y_test == 0, 0, 1)

    img_size_target, number_of_channels = x_train.shape[1], x_train.shape[-1]
    kernel_size = (5, 5) # TODO: should be in a config file
    initial_number_of_filters = 8 # TODO: should be in a config file

    input_layer = Input((img_size_target, img_size_target, number_of_channels))
    dropout_rate = 0.5
    output_layer = build_simpler_model(
        input_layer, initial_number_of_filters, kernel_size, dropout_rate
    )

    model = Model(input_layer, output_layer)

    METRICS = [
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]

    loss_function = 'binary_focal_crossentropy' # TODO: should be in a config file
    model.compile(
        loss=loss_function,
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        metrics=METRICS,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=30, # TODO: should be in a config file
        monitor="val_auc", # TODO: should be in a config file
        mode="max",
        restore_best_weights=True,
    )

    callbacks_list = [early_stop]

    BATCH_SIZE = 128
    EPOCHS = 200

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=0.2,
        )
    
    model.save("pipeline/outputs/model.h5")
    

if __name__ == "__main__":

    main()