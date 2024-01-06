"""
    neurogenetic_stacked_bar_features.py

# Description
This program reads a CSV file of neurogenetic disease phenotypes and creates a stacked bar chart
Neurogenetic disease are Charcot-Marie-Tooth, cerebellar ataxia, and hereditary spinal paraperesis
There are 31 different disease phenotypes

# Attribution
Author: Daniel B. Hier MD
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Change the working directory to where your data is located
os.chdir('/Users/danielhier/desktop/t_SNE')

# Define the path to the desired directory
directory_path = "/Users/danielhier/Desktop/t_SNE"

# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------

# Load your data into a Pandas DataFrame (replace 'neurogenetic.csv' with your data file)
neurogenetic = pd.read_csv(directory_path + '/neurogenetic.csv')
# Extract labels and features
labels = neurogenetic[['type', 'name']]
features = neurogenetic.iloc[:, 2:]

# Convert feature values to 0 or 1
features = features.applymap(lambda x: 1 if x > 0 else 0)
# Define custom colors for each group
# Concatenate the labels DataFrame and tsne_df along the columns axis
merged_df = pd.concat([labels, features], axis=1)
merged_df.to_csv('neurogenetic_diseases_binary_coded.csv')
# Now, merged_df contains four columns: 'type', 'name', 'tsne_dim1', and 'tsne_dim2'
# Sort the merged_df by the 'type' column

custom_colors = ['dodgerblue', 'olive', 'magenta']
# Define a list of perplexity values to loop through

#unique labels to colors using your custom colors
unique_labels = labels['type'].unique()
label_to_color = {label: custom_colors[i] for i, label in enumerate(unique_labels)}

# Map labels to custom colors
label_colors = [label_to_color[label] for label in labels['type']]
# Calculate the count of features for each disease type
feature_counts = features.groupby(labels['type']).sum()
# Transpose the feature_counts DataFrame to have features on the x-axis
feature_counts_transposed = feature_counts.T

# Sort the bars by the total count in descending order
sorted_features = feature_counts_transposed.sum(axis=1).sort_values(ascending=False).index
feature_counts_sorted = feature_counts_transposed.loc[sorted_features]

# Create a stacked bar chart
plt.figure(figsize=(12, 6))

# Define the color palette using your custom colors
colors = custom_colors

# Plot the stacked bars
feature_counts_sorted.plot(kind='bar', stacked=True, color=colors)

# Add labels and title
plt.xlabel('Features')
plt.ylabel('Counts')
plt.title('Phenotypes')

# Rename the legend by disease type
legend_labels = [plt.Rectangle((0,0),1,1, color=color) for color in custom_colors]
plt.legend(legend_labels, unique_labels, title='Disease', loc='upper right')

# Show the chart
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Ensure the labels fit within the figure boundaries
plt.savefig('stacked_bar_chart_neurogenetic.png', dpi=600, bbox_inches='tight')
plt.show()
