import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# List of 9 CSV files
csv_files = [
    # 'week8/top20_datasets/device1_top_20_features.csv',
    # 'week8/top20_datasets/device2_top_20_features.csv',
    # 'week8/top20_datasets/device3_top_20_features.csv',
    'week8/top20_datasets/device4_top_20_features.csv',
    'week8/top20_datasets/device5_top_20_features.csv',
    'week8/top20_datasets/device6_top_20_features.csv',
    'week8/top20_datasets/device7_top_20_features.csv',
    'week8/top20_datasets/device8_top_20_features.csv',
    'week8/top20_datasets/device9_top_20_features.csv'
]

# Loop through each CSV file
for i, csv_file in enumerate(csv_files, start=3):
    print(f"Processing file {i + 1} of 9: {csv_file}")

    # Load dataset from the current CSV file
    data = pd.read_csv(csv_file)

    # Separate features and labels
    X = data.drop(columns=['label'])
    y = data['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    random_forest_model = RandomForestClassifier()
    decision_tree_model = DecisionTreeClassifier()
    ada_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100)

    # Fit models to the training data
    random_forest_model.fit(X_train, y_train)
    decision_tree_model.fit(X_train, y_train)
    ada_model.fit(X_train, y_train)

    # Evaluate models on the test data
    rf_accuracy = accuracy_score(y_test, random_forest_model.predict(X_test))
    dt_accuracy = accuracy_score(y_test, decision_tree_model.predict(X_test))
    ada_accuracy = accuracy_score(y_test, ada_model.predict(X_test))

    print(f"Random Forest Accuracy: {rf_accuracy}")
    print(f"Decision Tree Accuracy: {dt_accuracy}")
    print(f"AdaBoost Accuracy: {ada_accuracy}")

    # Create a LIME explainer for each model
    rf_explainer = LimeTabularExplainer(X_train.values, mode='classification', feature_names=X_train.columns, class_names=['0', '1'])
    dt_explainer = LimeTabularExplainer(X_train.values, mode='classification', feature_names=X_train.columns, class_names=['0', '1'])
    ada_explainer = LimeTabularExplainer(X_train.values, mode='classification', feature_names=X_train.columns, class_names=['0', '1'])

    # Choose an instance from the test set for LIME explanations
    instance_idx = 0
    instance = X_test.iloc[[instance_idx]]
    true_class = y_test.iloc[instance_idx]

    # Get LIME explanations for each model
    rf_exp = rf_explainer.explain_instance(instance.values[0], random_forest_model.predict_proba, num_features=len(X.columns))
    dt_exp = dt_explainer.explain_instance(instance.values[0], decision_tree_model.predict_proba, num_features=len(X.columns))
    ada_exp = ada_explainer.explain_instance(instance.values[0], ada_model.predict_proba, num_features=len(X.columns))

    # Plot and save LIME feature importances for all models
    def save_lime_feature_importance_plot(exp, title, filename):
        weights = [exp.as_list(label=exp.available_labels()[0])[i][1] for i in range(len(exp.as_list(label=exp.available_labels()[0])))]
        features = [exp.as_list(label=exp.available_labels()[0])[i][0] for i in range(len(exp.as_list(label=exp.available_labels()[0])))]

        plt.figure(figsize=(8, 6))
        plt.bar(range(len(features)), weights)
        plt.xticks(range(len(features)), features, rotation=90)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    save_lime_feature_importance_plot(rf_exp, f'd{i + 1} Random Forest LIME Feature Importances', f'd{i + 1}_random_forest_lime_feature_importance.jpg')
    save_lime_feature_importance_plot(dt_exp, f'd{i + 1} Decision Tree LIME Feature Importances', f'd{i + 1}_decision_tree_lime_feature_importance.jpg')
    save_lime_feature_importance_plot(ada_exp, f'd{i + 1} AdaBoost LIME Feature Importances', f'd{i + 1}_ada_lime_feature_importance.jpg')
