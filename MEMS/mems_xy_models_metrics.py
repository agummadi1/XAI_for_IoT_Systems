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
data = pd.read_csv('xai_week1/datasets/mems_dataset.csv')

# Split the data into features (X) and labels (y)
X = data[['x', 'y']]
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
plt.savefig('mems_xy_metrics.jpg', format='jpg')

# Display the table
plt.show()
