#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Kmeans_Clustering_neurogenetic.py

# Attribution
Created on Thu Dec 7 20:48:12 2023
@author: danielhier
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
# import umap.umap_ as umap  # Corrected import statement
import umap     # Top-level umap import
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Declare the top-level data and output paths
data_dir = Path("data")
out_dir = Path("out", "kmeans")
out_dir.mkdir(exist_ok=True, parents=True)

# Point to the input data file
data_file = data_dir.joinpath("neurogenetic.csv")

# Name the output files
out_cluster_scatter = out_dir.joinpath("k_means_clustering_neurogenetic_scatter.png")
out_cluster_centroids = out_dir.joinpath("k_means_clustering_neurogenetic_centroids")
out_truth_scatter = out_dir.joinpath("k_means_ground_truth_neurogenetic_scatter.png")

# Set the DPI for each plot
DPI = 600

# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------


def main(display=True):

    # Load your data into a Pandas DataFrame
    neurogenetic = pd.read_csv(data_file)

    # Extract labels and features
    # labels = neurogenetic[['type', 'name']]
    _ = neurogenetic[['type', 'name']]
    features = neurogenetic.iloc[:, 2:]
    ground_truth_labels = neurogenetic['type']

    # Convert feature values to binary (0 or 1)
    # features = features.applymap(lambda x: 1 if x > 0 else 0)
    # features = features.map(lambda x: 1 if x > 0 else 0)
    try:
        # pandas API changed at 2.1.0
        features = features.map(lambda x: 1 if x > 0 else 0)
    except AttributeError:
        features = features.applymap(lambda x: 1 if x > 0 else 0)

    X = features

    # Perform K-Means clustering (set the number of clusters, e.g., 3)
    n_clusters = 3  # Adjust the number of clusters as needed
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Create a UMAP model with the desired number of components and other parameters
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.3,
        metric='euclidean',
        random_state=42,
    )

    # Fit the UMAP model to your binary feature matrix and transform it to the lower-dimensional space
    umap_result = umap_model.fit_transform(X)

    # Define a color mapping for cluster labels
    cluster_colors = {0: 'blue', 1: 'red', 2: 'saddlebrown'}

    # Map cluster labels to colors
    marker_colors = [cluster_colors[label] for label in cluster_labels]

    # Convert cluster labels to integers and add +1
    cluster_labels_int = [int(cluster_num) + 1 for cluster_num in cluster_labels]

    # Concatenate the cluster labels to the UMAP result using cluster_labels_int
    umap_result_2 = np.column_stack((umap_result, cluster_labels_int))

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    # scatter = plt.scatter(
    _ = plt.scatter(
        umap_result_2[:, 0],
        umap_result_2[:, 1], c=marker_colors
    )

    plt.xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    plt.ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')

    # plt.title('UMAP with K Means Clustering')

    # Create a custom legend mapping cluster numbers to colors
    legend_labels = [
        f'Cluster {cluster_num}' for cluster_num in range(1, max(cluster_labels_int) + 1)
    ]
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=14, label=label)
        for color, label in zip(cluster_colors.values(), legend_labels)
    ]

    # Add the custom legend to the plot
    plt.legend(handles=legend_handles, loc='upper left', fontsize=14)
    plt.savefig(out_cluster_scatter, dpi=DPI)

    plt.grid()
    _ = display and plt.show()

    # Create a UMAP model for ground_truth_labels
    umap_ground_truth = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.3,
        metric='euclidean',
        random_state=42,
    )

    # Fit the UMAP model to your binary feature matrix and transform it to the lower-dimensional space
    umap_result_ground_truth = umap_ground_truth.fit_transform(X)

    # Define a color mapping for ground truth labels
    # Adjust the colors as needed based on your ground truth labels
    ground_truth_colors = {'CMT': 'olive', 'CA': 'dodgerblue', 'HSP': 'magenta'}

    # Map ground truth labels to colors
    ground_truth_marker_colors = [
        ground_truth_colors[label]
        for label in ground_truth_labels
    ]

    # Create a scatter plot for ground truth labels
    plt.figure(figsize=(10, 6))
    # scatter_ground_truth = plt.scatter(
    _ = plt.scatter(
        umap_result_ground_truth[:, 0],
        umap_result_ground_truth[:, 1],
        c=ground_truth_marker_colors,
    )

    plt.xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    plt.ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')

    # plt.title('UMAP with Ground Truth Labels')

    # Create a custom legend mapping ground truth labels to colors
    # Adjust the labels as needed based on your ground truth labels
    ground_truth_legend_labels = ['CMT', 'CA', 'HSP']
    ground_truth_legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=14, label=label)
        for color, label in zip(ground_truth_colors.values(), ground_truth_legend_labels)
    ]

    # Add the custom legend to the plot
    plt.legend(handles=ground_truth_legend_handles, fontsize=14, loc='upper left')
    plt.grid()
    plt.savefig(out_truth_scatter, dpi=DPI)

    _ = display and plt.show()

    # Concatenate the cluster labels to the UMAP result with ground_truth labels
    umap_result_with_labels = np.column_stack((umap_result_2, ground_truth_labels))

    # Convert umap_result_with_labels to a DataFrame
    umap_df = pd.DataFrame(umap_result_with_labels, columns=[
                        'x_dim', 'y_dim', 'cluster_num', 'ground_truth_labels'])

    # Sort the DataFrame by cluster number
    umap_df = umap_df.sort_values(by='cluster_num')

    # Aggregate by mean to get centroids for each cluster
    cluster_centroids = umap_df.groupby('cluster_num').agg(
        {'x_dim': 'mean', 'y_dim': 'mean'}).reset_index()
    ground_truth_centroids = umap_df.groupby('ground_truth_labels').agg(
        {'x_dim': 'mean', 'y_dim': 'mean'}).reset_index()

    # Add columns from features to umap_df
    umap_df[
        ['atrophy', 'hyporeflexia', 'eye_movements', 'incoordination', 'hypertonia', 'hyperreflexia']
    ] = features[
        ['atrophy', 'hyporeflexia', 'eye_movements', 'incoordination', 'hypertonia', 'hyperreflexia']
    ]

    # Create a list to store the dictionaries for feature centroids
    feature_centroids_list = []

    # Iterate over the feature columns and calculate centroids
    for feature_column in ['atrophy', 'hyporeflexia', 'eye_movements', 'incoordination', 'hypertonia', 'hyperreflexia']:
        feature_mean_x = umap_df.loc[umap_df[feature_column] == 1, 'x_dim'].mean()
        feature_mean_y = umap_df.loc[umap_df[feature_column] == 1, 'y_dim'].mean()
        feature_centroids_list.append(
            {'Feature': feature_column, 'Mean_X': feature_mean_x, 'Mean_Y': feature_mean_y})

    # Create the feature_centroids DataFrame from the list
    feature_centroids = pd.DataFrame(feature_centroids_list)

    print(feature_centroids)

    plt.figure(figsize=(10, 6))
    for i, row in cluster_centroids.iterrows():
        x, y, cluster_num = row['x_dim'], row['y_dim'], int(
            row['cluster_num'])  # Convert to integer here
        plt.scatter(
            x, y, c=cluster_colors[cluster_num - 1],
            label=f'Cluster {cluster_num}', s=400
        )

        if cluster_num == 1:
            plt.annotate(
                f'Cluster {cluster_num}', (x, y), xytext=(-30, -60), textcoords='offset points', fontsize=14, fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='black')
            )
        elif cluster_num == 2:
            plt.annotate(
                f'Cluster {cluster_num}', (x, y), xytext=(-75, 10), textcoords='offset points', fontsize=14, fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='black')
            )
        else:
            plt.annotate(
                f'Cluster {cluster_num}', (x, y), xytext=(20, 70), textcoords='offset points', fontsize=14, fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='black')
            )

    # Plot ground_truth centroids with different colors
    for i, row in ground_truth_centroids.iterrows():
        x, y, ground_truth_label = row['x_dim'], row['y_dim'], row['ground_truth_labels']
        plt.scatter(
            x, y, marker='s',
            c=ground_truth_colors[ground_truth_label], label=f'ground_truth {ground_truth_label}', s=400
        )

        if ground_truth_label == 'CMT':
            plt.annotate(
                'Charcot-Marie-Tooth', (x, y), xytext=(70, 65), textcoords='offset points', fontsize=14, fontweight='bold', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black')
            )
        elif ground_truth_label == 'HSP':
            plt.annotate(
                'Hereditary Spastic Paraparesis', (x, y), xytext=(-95, 30), textcoords='offset points', fontsize=14, fontweight='bold', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black')
            )
        else:
            plt.annotate(
                'Cerebellar Ataxia', (x, y), xytext=(-40, -80), textcoords='offset points', fontsize=14, ha='center', fontweight='bold', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black')
            )

    # Plot feature centroids with grey markers and text annotations
    for i, row in feature_centroids.iterrows():
        x, y, feature_label = row['Mean_X'], row['Mean_Y'], row['Feature']
        plt.scatter(x, y, marker='o', c='black', s=100)

        if feature_label == 'atrophy':
            plt.annotate(
                feature_label, (x, y), xytext=(40, -40), textcoords='offset points', fontsize=12, ha='center', va='center', color='black',
                arrowprops=dict(arrowstyle='->', color='black')
            )
        elif feature_label == 'hyporeflexia':
            plt.annotate(
                feature_label, (x, y), xytext=(40, -40), textcoords='offset points', fontsize=12, ha='center', va='center', color='black',
                arrowprops=dict(arrowstyle='->', color='black')
            )
        elif feature_label == 'incoordination':
            plt.annotate(
                feature_label, (x, y), xytext=(-80, 0), textcoords='offset points', fontsize=12, ha='center', va='center', color='black',
                arrowprops=dict(arrowstyle='->', color='black')
            )
        elif feature_label == 'eye_movements':
            plt.annotate(
                feature_label, (x, y), xytext=(-80, 0), textcoords='offset points', fontsize=12, ha='center', va='center', color='black',
                arrowprops=dict(arrowstyle='->', color='black')
            )
        else:
            plt.annotate(
                feature_label, (x, y), xytext=(-40, 50), textcoords='offset points', fontsize=12, ha='center', va='center', color='black',
                arrowprops=dict(arrowstyle='->', color='black')
            )

    # ... Set x-axis and y-axis limits, labels, title, and save the plot ...

    # Add the legend for the ground_truth centroids
    # plt.legend(labels=ground_truth_centroids['ground_truth_labels'], loc='lower left')
    plt.grid()
    plt.xlabel('Mean_X', fontsize=12, fontweight='bold')
    plt.ylabel('Mean_Y', fontsize=12, fontweight='bold')

    # plt.title('K Means Clustering and Ground Truth Centroids')

    plt.savefig(out_cluster_centroids, dpi=DPI)

    _ = display and plt.show()

    return


if __name__ == "__main__":
    main()
