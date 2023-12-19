import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Load dataset
data = pd.read_csv('mems_dataset.csv')

# Extract the features (x, y, z) and labels
X = data[['x', 'y', 'z']]
y = data['label']

# Create a RandomForestClassifier
clf = RandomForestClassifier(random_state=42)

# Fit the classifier to data
clf.fit(X, y)

# Calculate feature importances using permutation importance
result = permutation_importance(clf, X, y, n_repeats=30, random_state=42)

# Get the feature importances and names
importances = result.importances_mean
feature_names = ['x', 'y', 'z']

# Create a bar plot to visualize feature importance
plt.barh(feature_names, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance using CEM')
plt.gca().invert_yaxis()  # Invert the y-axis for better visualization
plt.savefig('cem_feature_importance.jpg', format='jpg')  # Save the figure as a JPG
plt.show()
