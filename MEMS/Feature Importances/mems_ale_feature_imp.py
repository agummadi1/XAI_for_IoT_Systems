import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics import renderPDF
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fpdf import FPDF

# Load dataset
data = pd.read_csv('mems_dataset.csv')

# Split your data into training and testing sets:
X = data[['x', 'y', 'z']]  # Features 'x', 'y', 'z'
y = data['label']  # Target column 'label'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create a PDF report
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'ALE Plots and Explanations', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()


# ALE Plots for features 'x', 'y', and 'z':
features_to_plot = ['x', 'y', 'z']
pdf_elements = []


pdf = PDF()
pdf.add_page()

for feature_name in features_to_plot:
    plt.figure()
    for class_index, class_name in enumerate(y.unique()):
        ale_values = []  # To store ALE values
        grid_values = np.linspace(X_train[feature_name].min(), X_train[feature_name].max(), num=100)

        for grid_point in grid_values:
            X_temp = X_test.copy()
            X_temp[feature_name] = grid_point
            ale_value = model.predict_proba(X_temp)[:, class_index]
            ale_values.append(ale_value[0])

        # Plot ALE curve for the current class
        plt.plot(grid_values, ale_values, label=str(class_name))

    plt.xlabel(feature_name)
    plt.ylabel('ALE Value')
    plt.title(f'ALE Plot for {feature_name}')
    plt.legend()
    plt.savefig(f'ale_{feature_name}_plot.jpg')


    # Add the ALE plot and explanation to the PDF
    pdf.chapter_title(f'ALE Plot for {feature_name}')
    explanation = f"This is the ALE plot for feature '{feature_name}'. ALE (Accumulated Local Effects) plots " \
                  f"show how the model's predictions change as the feature values vary. The x-axis represents " \
                  f"the range of values for the feature, and the y-axis represents the ALE values. The plot " \
                  f"illustrates how the feature impacts the model's predictions."
    pdf.chapter_body(explanation)
    pdf.image(f'ale_{feature_name}_plot.jpg', x=10, y=None, w=190)

# Save the PDF report
pdf_filename = 'piezo_ale_report.pdf'
pdf.output(pdf_filename)