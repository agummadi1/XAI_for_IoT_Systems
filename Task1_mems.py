# #!/bin/bash

# #SBATCH -J Task1_job
# #SBATCH -p gpu
# #SBATCH -A r00381
# #SBATCH -o task1_output.txt
# #SBATCH -e task1_error.err
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=agummadi@iu.edu
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=5
# #SBATCH --gpus-per-node=5
# #SBATCH --time=06:00:00

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import tensorflow as tf
# import shap
# from sklearn.impute import SimpleImputer


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import shap
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the MEMS dataset
mems_data = pd.read_csv("xai_week1/datasets/mems_dataset.csv")

# Data Cleaning
imputer = SimpleImputer(strategy="mean")
mems_data.fillna(mems_data.mean(), inplace=True)

# Feature Selection
selected_mems_features = ["x", "y", "z", "label"]

mems_data = mems_data[selected_mems_features]

# Data Splitting
X = mems_data[["x", "y", "z"]]
y = mems_data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalization (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode the machine health condition labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Build a simple Keras neural network model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
model.fit(
    X_train,
    y_train_encoded,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test_encoded),
)

# SHAP Feature Importance Analysis using KernelExplainer
explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_test)

# Rename the class labels
class_labels = ["Normal", "Near-failure", "Failure"]

# Create a summary plot of SHAP feature importance with custom class labels
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=selected_mems_features[:-1],
    class_names=class_labels,
)

# Save the SHAP summary plot as a JPEG image
plt.savefig("task1_mems_output.jpg")  # Save the plot as "task1_mems_output.jpg"

