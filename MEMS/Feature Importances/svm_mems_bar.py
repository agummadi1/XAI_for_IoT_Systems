import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np

# Load dataset
data = pd.read_csv('xai_week1/datasets/mems_dataset.csv')

# Separate features (x, y, z) and labels
X = data[['x', 'y', 'z']]
y = data['label']

# Define SVM model
svm_model = SVC(kernel='linear', probability=True)  # Use a linear kernel for feature importances

# Fit SVM model to the data
svm_model.fit(X, y)

# Get feature importances (absolute values of coefficients)
svm_feature_importances = np.abs(svm_model.coef_).sum(axis=0)

# Plot and save feature importances for SVM
def save_feature_importance_plot(feature_importances, title, filename):
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(X.columns)), feature_importances)
    plt.xticks(range(len(X.columns)), X.columns, rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

save_feature_importance_plot(svm_feature_importances, 'SVM Feature Importances', 'svm_mems_bar_feature_importance.jpg')
