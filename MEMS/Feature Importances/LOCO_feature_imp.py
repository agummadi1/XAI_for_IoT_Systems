import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
import matplotlib.pyplot as plt

# Load your dataset from the CSV file
data = pd.read_csv("xai_week1/datasets/mems_dataset.csv")

# Separate features (X) and labels (y)
X = data[["x", "y", "z"]]
y = data["label"]

# Split the dataset into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Choose a specific data point for LOCO analysis (e.g., the first data point)
data_point_idx = 0
data_point = X_test.iloc[data_point_idx, :]
true_label = y_test.iloc[data_point_idx]

# Make a prediction for the chosen data point
original_prediction = rf_model.predict([data_point])[0]

# Create a copy of the model to modify for LOCO
loco_model = deepcopy(rf_model)

# Initialize an array to store prediction differences
prediction_differences = []

# Iterate through each feature and calculate prediction differences
for feature in X.columns:
    data_point_mean = X_test[feature].mean()
    data_point_copy = data_point.copy()
    data_point_copy[feature] = data_point_mean
    loco_prediction = loco_model.predict([data_point_copy])[0]
    prediction_difference = loco_prediction - original_prediction
    prediction_differences.append((feature, prediction_difference))

# Plot and save the prediction differences as a bar chart
features, differences = zip(*prediction_differences)
plt.figure(figsize=(10, 6))
plt.barh(features, differences)
plt.xlabel("Prediction Difference")
plt.ylabel("Feature")
plt.title("LOCO Prediction Differences")
plt.savefig("loco_predictions.jpg", bbox_inches="tight")
plt.show()