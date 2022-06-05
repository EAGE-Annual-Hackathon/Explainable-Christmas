import numpy as np
import shap
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
shap.explainers._deep.deep_tf.op_handlers[
    "Conv2DBackpropInput"
] = shap.explainers._deep.deep_tf.passthrough


def main():

    model = tf.keras.models.load_model("model.h5")

    flattened_model = tf.keras.models.Sequential()
    flattened_model.add(model)
    flattened_model.add(tf.keras.layers.Flatten())
    flattened_model.compile(loss="binary_focal_crossentropy")

    background_samples = np.load("pipeline/data/x_train_transformed.npy")[:, :64, :64, :]

    explanation_model = shap.DeepExplainer(flattened_model, background_samples)

    shap_values = explanation_model.shap_values(background_samples[0:1])

if __name__ == "__main__":
    main()