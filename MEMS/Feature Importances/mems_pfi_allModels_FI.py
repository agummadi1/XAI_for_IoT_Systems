import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# Load your dataset
data = pd.read_csv('xai_week1/datasets/mems_dataset.csv')

# Split the data into features (X) and labels (y)
X = data[['x', 'y', 'z']]
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary to store model names and the corresponding model objects
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'SVM': SVC(),
    'Deep Neural Network': MLPClassifier(random_state=42)
}

# Initialize a directory to store PFI summary plots
pfi_plot_dir = 'mems_pfi_plots/'

# Ensure the directory exists
import os
os.makedirs(pfi_plot_dir, exist_ok=True)

# Create bar graphs for PFI summary plots
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    # Calculate permutation feature importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
    
    # Get feature importance scores
    feature_importance = perm_importance.importances_mean
    
    # Create a bar plot to visualize feature importance
    plt.figure(figsize=(10, 6))
    feature_names = X.columns.tolist()
    plt.bar(feature_names, feature_importance, label=model_name)
    plt.xlabel('Features')
    plt.ylabel('Importance Score (Drop in Accuracy)')
    plt.title(f'PFI Feature Importance for {model_name}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the PFI summary bar graph as a JPG image
    plt.savefig(os.path.join(pfi_plot_dir, f'mems_pfi_summary_{model_name}.jpg'))
    plt.close()