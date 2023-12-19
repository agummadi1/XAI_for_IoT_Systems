import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np

# Load dataset
data = pd.read_csv('xai_week1/datasets/mems_dataset.csv')

# Separate features (x, y, z) and labels
X = data[['x', 'y', 'z']]
y = data['label']

# Define models
random_forest_model = RandomForestClassifier()
decision_tree_model = DecisionTreeClassifier()
logistic_regression_model = LogisticRegression(max_iter=10000)
# svm_model = SVC(probability=True)  # Use probability=True for feature importances with SVM
# mlp_model = MLPClassifier(max_iter=1000)
ada_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100)

# Fit models to the data
random_forest_model.fit(X, y)
decision_tree_model.fit(X, y)
logistic_regression_model.fit(X, y)
# svm_model.fit(X, y)
# mlp_model.fit(X, y)
ada_model.fit(X, y)

# Get feature importances
# Error : 'KNeighborsClassifier' object has no attribute 'feature_importances_'
random_forest_feature_importances = random_forest_model.feature_importances_
decision_tree_feature_importances = decision_tree_model.feature_importances_
# svm_feature_importances = np.abs(svm_model.coef_).sum(axis=0)  # Use absolute values for SVM
# mlp_feature_importances = np.abs(mlp_model.coefs_[0]).sum(axis=0)  # Use absolute values for MLP
ada_feature_importances = ada_model.feature_importances_

# Plot and save feature importances for all models
def save_feature_importance_plot(feature_importances, title, filename):
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(X.columns)), feature_importances)
    plt.xticks(range(len(X.columns)), X.columns, rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

save_feature_importance_plot(random_forest_feature_importances, 'Random Forest Feature Importances', 'random_forest_feature_importance.jpg')
save_feature_importance_plot(decision_tree_feature_importances, 'Decision Tree Feature Importances', 'decision_tree_feature_importance.jpg')
# save_feature_importance_plot(svm_feature_importances, 'SVM Feature Importances', 'svm_feature_importance.jpg')
# save_feature_importance_plot(mlp_feature_importances, 'MLP Feature Importances', 'mlp_feature_importance.jpg')
save_feature_importance_plot(ada_feature_importances, 'AdaBoost Feature Importances', 'ada_feature_importance.jpg')