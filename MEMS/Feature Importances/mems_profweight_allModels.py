import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load your dataset from "mems_dataset.csv"
df = pd.read_csv("xai_week1/datasets/mems_dataset.csv")

X = df[['x', 'y', 'z']]
y = df['label']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a dictionary to store feature importance for different models
feature_importance = {}

# Define and evaluate feature importance for different models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(random_state=42), random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "DNN": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

feature_names = X.columns.tolist()  # Define feature names here

for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    # Store the original model's performance
    original_accuracy = model.score(X_test, y_test)
    
    # Iterate through each feature to assess its importance
    for feature in X.columns:
        # Create a perturbed dataset with the feature shuffled or modified
        X_perturbed = X_test.copy()
        X_perturbed[feature] = np.random.permutation(X_perturbed[feature])

        # Calculate the accuracy change due to the perturbation
        perturbed_accuracy = model.score(X_perturbed, y_test)
        importance_score = original_accuracy - perturbed_accuracy

        if feature not in feature_importance:
            feature_importance[feature] = {}
        feature_importance[feature][model_name] = importance_score

# Create separate bar plots to visualize feature importance for each model
for model_name in models.keys():
    plt.figure(figsize=(8, 6))
    weights = [feature_importance[feature][model_name] for feature in feature_names]
    plt.bar(feature_names, weights)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title(f'Feature Importance Using ProfWeight-Inspired Method ({model_name})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'mems_{model_name}_feature_importance.jpg')
    plt.show()
