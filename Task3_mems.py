import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import shap
from sklearn.impute import SimpleImputer
from tabulate import tabulate

# Load the MEMS dataset
mems_data = pd.read_csv('xai_week1/datasets/mems_dataset.csv')

# Data Cleaning
imputer = SimpleImputer(strategy='mean')
mems_data.fillna(mems_data.mean(), inplace=True)

# Feature Selection
selected_mems_features = ['x', 'y', 'z', 'label']

mems_data = mems_data[selected_mems_features]

# Data Splitting
X = mems_data[['x', 'y', 'z']]
y = mems_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode the machine health condition labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Build a simple Keras neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),  # Input layer with 3 features
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer with 64 neurons and ReLU activation
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 classes and softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))

# SHAP Feature Importance Analysis using KernelExplainer
explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_test)

# Rename the class labels
class_labels = ['Normal', 'Near-failure', 'Failure']

# Reshape the shap_values array
shap_values_reshaped = shap_values[0]  # Assuming binary classification, choose the relevant class

# Calculate the mean absolute SHAP values for each feature per label
shap_df = pd.DataFrame(shap_values_reshaped, columns=selected_mems_features[:-1])
shap_df['label'] = y_test.values  # Use the true labels from your test data
mean_abs_shap_per_label = shap_df.groupby('label').mean().abs()

# Display the mean absolute SHAP values per label using tabulate
table_headers = ["Feature", "Normal", "Near-failure", "Failure"]
table_data = []
for feature in selected_mems_features[:-1]:
    row = [feature] + mean_abs_shap_per_label[feature].tolist()
    table_data.append(row)

print(tabulate(table_data, headers=table_headers, tablefmt='grid'))
