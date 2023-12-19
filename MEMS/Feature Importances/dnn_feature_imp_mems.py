import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# Load your dataset
data = pd.read_csv('xai_week1/datasets/mems_dataset.csv')

# Assuming your dataset has columns 'x', 'y', 'z', and 'labels'
X = data[['x', 'y', 'z']]
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a neural network (DNN) model
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=0)
model.fit(X_train, y_train)

# Calculate baseline accuracy
baseline_accuracy = accuracy_score(y_test, model.predict(X_test))

# Calculate permutation importance for each feature
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)

# Get the importance scores for each feature
importance_scores = result.importances_mean

# Print the importance scores for each feature
for feature, importance in zip(['x', 'y', 'z'], importance_scores):
    print(f"{feature} Importance: {importance:.4f}")

# You can also rank the features by importance
sorted_feature_importance = np.argsort(importance_scores)[::-1]
print("Feature ranking:", [(['x', 'y', 'z'][i], importance_scores[i]) for i in sorted_feature_importance])

# Save the output as a JPG file
plt.figure(figsize=(8, 6))
plt.bar(['x', 'y', 'z'], importance_scores)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.savefig('dnn_feature_importance_mems.jpg')
plt.show()
