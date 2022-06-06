# Explainable-Christmas


## The Project
We used the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge) data along with a U-Net model trained from scratch using Tensorflow's library. Additionally, we created three seismic attributes from the seismic data to use as new features while training the model. Finally, we used [Shap library](https://github.com/slundberg/shap) to calculate SHAP values and obtain a measure of how important each feature is. This allowed to evaluate possible biases and if the new features were contributing to the predictions and not being underused.


## Training and Explaining Pipelines
To run the entire pipeline using MLFlow, use the following command:
`mlflow run --entry-point explain --no-conda .`
This will process the data again, segregate then explain the predictions of a few samples.
The --no-conda argument will make the current environment the one that is going to be used to run this command.
The entry points and their commands are described in the MLproject file.

For instance, to only segregate and train you can use the following command:
`mlflow run --entry-point segregate_train --no-conda .`

Metrics and artifacts, such as the model, of each experiment are automatically logged with MLFlow.
Run the following command to observe the details of each experiment:
`mlflow ui`

## Team Members
Edwin Brown and Marcos Jacinto