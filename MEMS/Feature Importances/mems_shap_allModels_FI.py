import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
import shap

# Load your dataset
data = pd.read_csv('xai_week1/datasets/mems_dataset.csv')

# Split the data into features (X) and labels (y)
X = data[['x', 'y', 'z']]
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary to store model names and the corresponding model objects
models = {
    # 'Decision Tree': DecisionTreeClassifier(),
    # 'Random Forest': RandomForestClassifier(),
    # 'AdaBoost': AdaBoostClassifier(),
    'SVM': SVC()
    # ,
    # 'Deep Neural Network': MLPClassifier(random_state=42)
}

# Initialize a directory to store SHAP summary plots
shap_plot_dir = 'mems_shap_plots/'

# Ensure the directory exists
import os
os.makedirs(shap_plot_dir, exist_ok=True)

# Create bar graphs for SHAP summary plots
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    # Use SHAP to calculate feature importance with KernelExplainer
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test, check_additivity=False)  # Disable additivity check
    
    # Plot and save the SHAP summary plot as a bar graph
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.title(f'SHAP Summary Bar Plot for {model_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the SHAP summary bar graph as a JPG image
    plt.savefig(os.path.join(shap_plot_dir, f'mems_shap_summary_bar_{model_name}.jpg'))
    plt.close()
