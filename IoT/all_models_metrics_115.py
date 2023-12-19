import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

# Load your dataset
data = pd.read_csv('iot_datasets/device1_combined.csv')

# Modify this list with the names of your 115 features
feature_names = ['MI_dir_L5_weight', 'MI_dir_L5_mean', 'MI_dir_L5_variance', 'MI_dir_L3_weight', 'MI_dir_L3_mean', 'MI_dir_L3_variance', 'MI_dir_L1_weight', 'MI_dir_L1_mean', 'MI_dir_L1_variance', 'MI_dir_L0.1_weight', 'MI_dir_L0.1_mean', 'MI_dir_L0.1_variance', 'MI_dir_L0.01_weight', 'MI_dir_L0.01_mean', 'MI_dir_L0.01_variance', 'H_L5_weight', 'H_L5_mean', 'H_L5_variance', 'H_L3_weight', 'H_L3_mean', 'H_L3_variance', 'H_L1_weight', 'H_L1_mean', 'H_L1_variance', 'H_L0.1_weight', 'H_L0.1_mean', 'H_L0.1_variance', 'H_L0.01_weight', 'H_L0.01_mean', 'H_L0.01_variance', 'HH_L5_weight', 'HH_L5_mean', 'HH_L5_std', 'HH_L5_magnitude', 'HH_L5_radius', 'HH_L5_covariance', 'HH_L5_pcc', 'HH_L3_weight', 'HH_L3_mean', 'HH_L3_std', 'HH_L3_magnitude', 'HH_L3_radius', 'HH_L3_covariance', 'HH_L3_pcc', 'HH_L1_weight', 'HH_L1_mean', 'HH_L1_std', 'HH_L1_magnitude', 'HH_L1_radius', 'HH_L1_covariance', 'HH_L1_pcc', 'HH_L0.1_weight', 'HH_L0.1_mean', 'HH_L0.1_std', 'HH_L0.1_magnitude', 'HH_L0.1_radius', 'HH_L0.1_covariance', 'HH_L0.1_pcc', 'HH_L0.01_weight', 'HH_L0.01_mean', 'HH_L0.01_std', 'HH_L0.01_magnitude', 'HH_L0.01_radius', 'HH_L0.01_covariance', 'HH_L0.01_pcc', 'HH_jit_L5_weight', 'HH_jit_L5_mean', 'HH_jit_L5_variance', 'HH_jit_L3_weight', 'HH_jit_L3_mean', 'HH_jit_L3_variance', 'HH_jit_L1_weight', 'HH_jit_L1_mean', 'HH_jit_L1_variance', 'HH_jit_L0.1_weight', 'HH_jit_L0.1_mean', 'HH_jit_L0.1_variance', 'HH_jit_L0.01_weight', 'HH_jit_L0.01_mean', 'HH_jit_L0.01_variance', 'HpHp_L5_weight', 'HpHp_L5_mean', 'HpHp_L5_std', 'HpHp_L5_magnitude', 'HpHp_L5_radius', 'HpHp_L5_covariance', 'HpHp_L5_pcc', 'HpHp_L3_weight', 'HpHp_L3_mean', 'HpHp_L3_std', 'HpHp_L3_magnitude', 'HpHp_L3_radius', 'HpHp_L3_covariance', 'HpHp_L3_pcc', 'HpHp_L1_weight', 'HpHp_L1_mean', 'HpHp_L1_std', 'HpHp_L1_magnitude', 'HpHp_L1_radius', 'HpHp_L1_covariance', 'HpHp_L1_pcc', 'HpHp_L0.1_weight', 'HpHp_L0.1_mean', 'HpHp_L0.1_std', 'HpHp_L0.1_magnitude', 'HpHp_L0.1_radius', 'HpHp_L0.1_covariance', 'HpHp_L0.1_pcc', 'HpHp_L0.01_weight', 'HpHp_L0.01_mean', 'HpHp_L0.01_std', 'HpHp_L0.01_magnitude', 'HpHp_L0.01_radius', 'HpHp_L0.01_covariance', 'HpHp_L0.01_pcc']

# Split the data into features (X) and labels (y)
X = data[feature_names]
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

# Initialize an empty DataFrame to store performance metrics
metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Iterate through the models and calculate performance metrics
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics_df = metrics_df.append({'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}, ignore_index=True)

# Print the performance metrics table
print(metrics_df)

# Create a table image using matplotlib
plt.figure(figsize=(10, 6))
plt.axis('off')  # Turn off axis
plt.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center', colColours=['#f2f2f2']*len(metrics_df.columns))
plt.tight_layout()

# Save the table as a JPG image
plt.savefig('device1_metrics.jpg', format='jpg')

# Display the table
plt.show()
