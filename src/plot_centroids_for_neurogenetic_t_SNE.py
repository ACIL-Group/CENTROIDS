"""
    plot_centroids_for_neurogenetic_t_SNE.py

# Attribution
Author: Daniel B. Hier MD
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

from adjustText import adjust_text
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from adjustText import adjust_text
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Declare the top-level data and output paths
data_dir = Path("data")
out_dir = Path("out", "centroids")
out_dir.mkdir(exist_ok=True, parents=True)

# Point to the data file
data_file = data_dir.joinpath("neurogenetic.csv")

# Name the output files
out_binary_coded_csv = out_dir.joinpath("neurogenetic_diseases_binary_coded.csv")
out_coordinate_csv = out_dir.joinpath("neurogenetic_diseases_coordinate.csv")
out_centroids = out_dir.joinpath('t_SNE_centroids_neurogenetic')

# Set the DPI for each plot
DPI = 600

# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------

################################
# Read in Data for t-SNE plotting.
################################

# Load your data into a Pandas DataFrame
neurogenetic = pd.read_csv(data_file)

# Extract labels and features
labels = neurogenetic[['type', 'name']]
features = neurogenetic.iloc[:, 2:]
# Convert feature values to 0 or 1
features = features.applymap(lambda x: 1 if x > 0 else 0)
# Define custom colors for each group
# Concatenate the labels DataFrame and tsne_df along the columns axis
merged_df = pd.concat([labels, features], axis=1)
merged_df.to_csv(out_binary_coded_csv)
# Now, merged_df contains four columns: 'type', 'name', 'tsne_dim1', and 'tsne_dim2'
# Sort the merged_df by the 'type' column

########################################################################################################
# CCreat first t-SNE plot. It is an unlabeled swarm of markers representing three neurogenetic diseases#
########################################################################################################

custom_colors = ['dodgerblue', 'olive', 'magenta']
# Define a list of perplexity values to loop through
perplexity = 50
# Initialize an empty DataFrame to store the mean coordinates for each feature
features_means = pd.DataFrame(columns=['Feature', 'Mean_X', 'Mean_Y'])
initialize = 'pca'
# Create a t-SNE model with the current perplexity value
tsne = TSNE(n_components=2, random_state=42, init=initialize, perplexity=perplexity, method='exact', early_exaggeration=10, n_iter=1000, metric='euclidean')
tsne_result = tsne.fit_transform(features)
# Create a mapping from unique labels to colors using your custom colors
unique_labels = labels['type'].unique()
label_to_color = {label: custom_colors[i] for i, label in enumerate(unique_labels)}

# Map labels to custom colors
label_colors = [label_to_color[label] for label in labels['type']]

#Create a scatter plot for merged_features_means
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

# Create a scatter plot with custom colors for each group
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=label_colors)

# Create a legend using dummy points with a smaller font size and smaller markers
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=14) for label, color in label_to_color.items()]

# Add the legend to the plot
legend = plt.legend(handles=legend_handles, labels=label_to_color.keys(), loc='upper right')

# Adjust the legend box properties
legend.get_frame().set_edgecolor('0.5')  # Set the edge color of the legend box
legend.get_frame().set_linewidth(0.5)   # Set the edge linewidth of the legend box

#plt.title(f't-SNE plot (Perplexity={perplexity}, Initialization={initialize})')
plt.xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
plt.grid(True)
plot_type='t_SNE_'+str(perplexity)+'_'+str(initialize)
group_type='neurogenetic'
# filename = plot_type + '_' + group_type +"_no_annotations"
filename = out_dir.joinpath(plot_type + '_' + group_type +"_no_annotations")
# Save the plot as an image file with a higher dpi value and adjusted legend position
plt.savefig(filename, dpi=DPI, bbox_inches='tight', bbox_extra_artists=[legend])
plt.show()

#--------------------------------------End of First Plot------------------------------------
#--------------------------------------Begin Annotating Feature Centroids for Second Plot---
custom_colors = ['dodgerblue', 'olive', 'magenta']
# Define a list of perplexity values to loop through
perplexity = 50
# Initialize an empty DataFrame to store the mean coordinates for each feature
features_means = pd.DataFrame(columns=['Feature', 'Mean_X', 'Mean_Y'])
initialize = 'pca'
# Create a t-SNE model with the current perplexity value
tsne = TSNE(n_components=2, random_state=42, init=initialize, perplexity=perplexity, method='exact', early_exaggeration=10, n_iter=1000, metric='euclidean')
tsne_result = tsne.fit_transform(features)

# Create a mapping from unique labels to colors using your custom colors
unique_labels = labels['type'].unique()
label_to_color = {label: custom_colors[i] for i, label in enumerate(unique_labels)}
# Map labels to custom colors
label_colors = [label_to_color[label] for label in labels['type']]
#Create a scatter plot for merged_features_means

plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
# Create a scatter plot with custom colors for each group

# Create a scatter plot with custom colors for each group
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=label_colors)

# Create a legend using dummy points with a smaller font size and smaller markers
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=14) for label, color in label_to_color.items()]

# Add the legend to the plot
legend = plt.legend(handles=legend_handles, labels=label_to_color.keys(), loc='upper right')

# Adjust the legend box properties
legend.get_frame().set_edgecolor('0.5')  # Set the edge color of the legend box
legend.get_frame().set_linewidth(0.5)   # Set the edge linewidth of the legend box

#plt.title(f't-SNE plot (Perplexity={perplexity}, Initialization={initialize})')
plt.xlabel('t-SNE Dimension 1',fontsize=12, fontweight='bold')
plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')

# List of features to consider
features_to_consider = ['hyperreflexia','sensory','hypertonia', 'hyporeflexia','eye_movements','incoordination','atrophy','tremor','weakness']

# Initialize a list to store annotation texts
annotation_texts = []

# Loop through each feature
for feature in features_to_consider:
    # Filter the rows in the neurogenetic DataFrame where the specified feature has a value of 1
    filtered_neurogenetic = neurogenetic[neurogenetic[feature] == 1]

    # Check if the filtered data is empty
    if not filtered_neurogenetic.empty:
        # Calculate the mean x and y coordinates in tsne_results for the filtered rows
        mean_x = np.mean(tsne_result[filtered_neurogenetic.index, 0])
        mean_y = np.mean(tsne_result[filtered_neurogenetic.index, 1])

        # Append the feature and mean coordinates to the features_means DataFrame
        features_means = pd.concat([features_means,
                                    pd.DataFrame({'Feature': [feature], 'Mean_X': [mean_x], 'Mean_Y': [mean_y]})],
                                   ignore_index=True)

        # Add the annotation text to the list
        annotation_texts.append(
            plt.text(mean_x, mean_y, feature, fontsize=14, fontweight='bold', ha='center'))

# Use adjust_text to automatically adjust the annotations to avoid overlap with straight-line arrows
adjust_text(
    annotation_texts,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.0"),
    # expand_text=(1.2, 1.2)
)

print(features_means)
plt.grid(True)
plot_type='t_SNE_'+str(perplexity)+'_'+str(initialize)
group_type='neurogenetic'
# filename =plot_type + '_' + group_type
filename = out_dir.joinpath(plot_type + '_' + group_type)
# Save the plot as an image file with a higher dpi value and adjusted legend position
plt.savefig(filename, dpi=DPI, bbox_inches='tight', bbox_extra_artists=[legend])

plt.show()
# Assuming you have a DataFrame named 'df'
# features_means.to_csv(filename + '.csv', index=False)  # Use index=False to exclude row numbers in the output
features_means.to_csv(filename.with_suffix('.csv'), index=False)  # Use index=False to exclude row numbers in the output

#--------------------------------------End Second Plot with Feature Centroids-------------------------------
#--------------------------------------Begin Third Plot with Class Swarm replaced by Class Centroids-------
# Create a DataFrame from tsne_result with appropriate column names
# Clear the current figure to remove all previous markers and elements
plt.clf()

# Now, you can proceed to create your third plot

tsne_df = pd.DataFrame(tsne_result, columns=['tsne_dim1', 'tsne_dim2'])

# Concatenate the labels DataFrame and tsne_df along the columns axis
merged_df = pd.concat([labels, tsne_df], axis=1)
merged_df.to_csv(out_coordinate_csv)

# Now, merged_df contains four columns: 'type', 'name', 'tsne_dim1', and 'tsne_dim2'
# Sort the merged_df by the 'type' column
sorted_merged_df = merged_df.sort_values(by='type')

# Group by 'type' and calculate the mean for 'tsne_dim1' and 'tsne_dim2'
mean_df = sorted_merged_df.groupby('type')[['tsne_dim1', 'tsne_dim2']].mean().reset_index()

# The resulting mean_df will have 'type', 'tsne_dim1_mean', and 'tsne_dim2_mean' columns
mean_df.columns = ['type', 'tsne_dim1_mean', 'tsne_dim2_mean']
# Change column headings for mean_df
mean_df.columns = ['Feature', 'Mean_X', 'Mean_Y']

# Merge mean_df with features_means by concatenating them
merged_features_means = pd.concat([features_means, mean_df], ignore_index=True)

# Now merged_features_means will have the same columns: 'Feature', 'Mean_X', and 'Mean_Y'

# Define a dictionary to map feature labels to custom colors
feature_colors = {'CA': 'dodgerblue', 'CMT': 'olive', 'HSP': 'magenta'}

# Create a scatter plot for merged_features_means
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

# Create a scatter plot with custom colors for each group
#plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=label_colors)

# Create a legend using dummy points with a smaller font size and smaller markers
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=14) for label, color in label_to_color.items()]

# Add the legend to the plot
legend = plt.legend(handles=legend_handles, labels=label_to_color.keys(), loc='upper right')

# Adjust the legend box properties
legend.get_frame().set_edgecolor('0.5')  # Set the edge color of the legend box
legend.get_frame().set_linewidth(0.5)   # Set the edge linewidth of the legend box

# Loop through each row in merged_features_means
for index, row in merged_features_means.iterrows():
    label = row['Feature']
    if label in feature_colors:
        color = feature_colors[label]  # Use the custom color if available
    else:
        color = 'gray'  # Use gray for other labels

    if label == 'CA':
        # For "cerebellar ataxia," adjust the label position and arrow
        label = r'$\bf{cerebellar\  ataxia}$'
        plt.annotate(label, (row['Mean_X'], row['Mean_Y']), xytext=(60, -90), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"), ha='center',fontsize=14)
        marker_size = 400  # Set the marker size to 20
        plt.scatter(row['Mean_X'], row['Mean_Y'], s=marker_size, c=color)
    elif label == 'HSP':
        # For "spastic paraparesis," adjust the label position and arrow
        label = r'$\bf{hereditary\  spastic\  paraparesis}$'
        plt.annotate(label, (row['Mean_X'], row['Mean_Y']), xytext=(100, 30), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"), ha='center',fontsize=14)
        marker_size = 400  # Set the marker size to 20
        plt.scatter(row['Mean_X'], row['Mean_Y'], s=marker_size, c=color)
    elif label == 'CMT':
        # For "Charcot-Marie-Tooth," adjust the label position and arrow
        label = r'$\bf{Charcot-Marie-Tooth}$'
        plt.annotate(label, (row['Mean_X'], row['Mean_Y']), xytext=(-80, -80), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"), ha='center',fontsize=14)
        marker_size = 400  # Set the marker size to 20
        plt.scatter(row['Mean_X'], row['Mean_Y'], s=marker_size, c=color)
    elif label == 'incoordination':
            # For "incoordination," adjust the label position and arrow
            plt.annotate(label, (row['Mean_X'], row['Mean_Y']), xytext=(50, -20), textcoords="offset points",
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"), ha='center',fontsize=14)
            marker_size = 50  # Set the marker size to 20
            plt.scatter(row['Mean_X'], row['Mean_Y'], s=marker_size, c=color)
    elif label == 'eye_movements':
                label = 'eye movements'
                # For "incoordination," adjust the label position and arrow
                plt.annotate(label, (row['Mean_X'], row['Mean_Y']), xytext=(60, -20), textcoords="offset points",
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"), ha='center',fontsize=14)
                marker_size = 50  # Set the marker size to 20
                plt.scatter(row['Mean_X'], row['Mean_Y'], s=marker_size, c=color)
    elif label == 'tremor':
                  label = 'tremor'
                  # For "incoordination," adjust the label position and arrow
                  plt.annotate(label, (row['Mean_X'], row['Mean_Y']), xytext=(60, -20), textcoords="offset points",
                               arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"), ha='center',fontsize=14)
                  marker_size = 50  # Set the marker size to 20
                  plt.scatter(row['Mean_X'], row['Mean_Y'], s=marker_size, c=color)

    else:
        # For other labels, use the same settings as before
        plt.annotate(label, (row['Mean_X'], row['Mean_Y']), xytext=(-50, 10), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"), ha='center', fontsize=14)
        marker_size = 50  # Set the marker size to 20
        plt.scatter(row['Mean_X'], row['Mean_Y'], s=marker_size, c=color)


# Add labels and title
plt.xlabel('Mean_X', fontsize=12, fontweight='bold')
plt.ylabel('Mean_Y',  fontsize=12, fontweight='bold')
#plt.title('Plot of Disease and Feature Centroids for Neurogenetic Diseases')
#filename='centroids_neurogenetic_t_sne_'
# Show the plot
plt.grid(True)
# plt.savefig('t_SNE_+centroids_'+'neurogenetic', dpi=DPI, bbox_inches='tight', bbox_extra_artists=[legend])
plt.savefig(out_centroids, dpi=DPI, bbox_inches='tight', bbox_extra_artists=[legend])
plt.show()
