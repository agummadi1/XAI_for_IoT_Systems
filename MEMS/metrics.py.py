import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from tabulate import tabulate

def load_data(file_path, selected_features):
    data = pd.read_csv(file_path)
    data.fillna(data.mean(), inplace=True)
    data = data[selected_features]
    return data

def preprocess_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return X_train, X_test, y_train_encoded, y_test_encoded, label_encoder

def evaluate_model(model, X_test, y_test_encoded):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, predictions)
    bacc = balanced_accuracy_score(y_test_encoded, predictions)
    mcc = matthews_corrcoef(y_test_encoded, predictions)

    return predictions, accuracy, bacc, mcc

def calculate_auc_roc(model, X_test, y_test_encoded, label_encoder):
    probabilities = model.predict_proba(X_test)
    auc_roc_values = []

    for class_idx in range(len(label_encoder.classes_)):
        class_probabilities = probabilities[:, class_idx]
        class_indicator = (y_test_encoded == class_idx).astype(int)
        class_auc_roc = roc_auc_score(class_indicator, class_probabilities)
        auc_roc_values.append(class_auc_roc)

    avg_auc_roc = sum(auc_roc_values) / len(auc_roc_values)
    return avg_auc_roc

def main():
    # Load data
    selected_features = ['x', 'y', 'z', 'label']
    piezoelectric_data = load_data('xai_week1/datasets/piezoelectric_dataset_1.2.csv', selected_features)

    # Preprocess data
    X_train, X_test, y_train_encoded, y_test_encoded, label_encoder = preprocess_data(piezoelectric_data, 'label')

    # Initialize models
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    ada_classifier = AdaBoostClassifier(n_estimators=50)
    svm_classifier = SVC(kernel='linear', C=1.0, probability=True)
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)

    # Train models
    knn_classifier.fit(X_train, y_train_encoded)
    ada_classifier.fit(X_train, y_train_encoded)
    svm_classifier.fit(X_train, y_train_encoded)
    mlp_classifier.fit(X_train, y_train_encoded)

    # Evaluate models
    results = []
    models = [
        ("K-Nearest Neighbors (KNN)", knn_classifier),
        ("Adaptive Boosting (ADA)", ada_classifier),
        ("Support Vector Machine (SVM)", svm_classifier),
        ("Multi-Layer Perceptron (MLP)", mlp_classifier),
    ]

    for model_name, model in models:
        predictions, accuracy, bacc, mcc = evaluate_model(model, X_test, y_test_encoded)
        auc_roc = calculate_auc_roc(model, X_test, y_test_encoded, label_encoder)
        report = classification_report(y_test_encoded, predictions, output_dict=True)

        results.append([model_name, accuracy, bacc, mcc, auc_roc, report])

    # Display results in a table
    table_headers = ["Model", "Accuracy", "Balanced Accuracy", "MCC", "AUC-ROC", "Precision", "Recall", "F1-Score"]
    table_data = [
        [model_name, accuracy, bacc, mcc, auc_roc, report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']]
        for model_name, accuracy, bacc, mcc, auc_roc, report in results
    ]

    print(tabulate(table_data, headers=table_headers, tablefmt='grid'))

if __name__ == "__main__":
    main()
