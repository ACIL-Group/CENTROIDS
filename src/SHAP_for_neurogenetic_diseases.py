#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    SHAP_for_neurogenetic_diseases.py

# Description
This program fits a xgboost tree to the neurogenetic data and then uses SHAP to find the 10 most influential phenotype features.

# Attribution
Created on Wed Dec 6 09:49:21 2023
@author: danielhier
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import shap
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Declare the top-level data and output paths
data_dir = Path("data")
out_dir = Path("out", "shap")
out_dir.mkdir(exist_ok=True, parents=True)

# Point to the data file
data_file = data_dir.joinpath("neurogenetic.csv")

# Name the output files
out_shap = out_dir.joinpath("shap_summary_plot.png")

# Set the DPI for each plot
DPI = 600

# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------

# Load your data into a Pandas DataFrame
neurogenetic = pd.read_csv(data_file)

# Extract labels and features
labels = neurogenetic[['type', 'name']]
features = neurogenetic.iloc[:, 2:]

# Convert feature values to 0 or 1
# features = features.applymap(lambda x: 1 if x > 0 else 0)
features = features.map(lambda x: 1 if x > 0 else 0)

# Concatenate 'features' and 'labels' along the columns axis
df_neurogenetic = pd.concat([labels, features], axis=1)

# Define the features and labels
X = df_neurogenetic.iloc[:, 2:]
y = df_neurogenetic['type']

# Encode the text labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
clf = xgb.XGBClassifier()

# Fit the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate class probabilities for the test data
class_probs = clf.predict_proba(X_test)

# Create a SHAP explainer object with the XGBoost model
explainer = shap.Explainer(clf, X_train)

# Calculate SHAP values for your test data
shap_values = explainer.shap_values(X_test)

# Create a mapping from encoded class labels to disease names
class_label_to_disease = {label: disease for label, disease in zip(label_encoder.classes_, df_neurogenetic['type'])}
class_label_to_disease[0] = 'CA'
class_label_to_disease[1] = 'CMT'
class_label_to_disease[2] = 'HSP'
# Ensure all class labels in clf.classes_ are in the mapping dictionary
for label in clf.classes_:
    if label not in class_label_to_disease:
        # Handle the case where the label is not in the mapping (you can assign a default disease name)
        class_label_to_disease[label] = "Unknown Disease"

# Get class names for the legend
class_names = [class_label_to_disease[label] for label in clf.classes_]

# Summarize the feature importance with class names in the legend
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns,
    class_names=class_names,
    max_display=10,
    show=False,
)

# Save the plot to a file
plt.tight_layout()
plt.savefig(out_shap, dpi=DPI)  # Adjust the file name and DPI as needed

# Show the plot
plt.show()
