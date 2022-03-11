from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def cluster(coordinates, debug=False):
    """
    Given a set of GPS co-ordinates, find the most likely location of the target
    """
    #dbscan configuration
    epsilon = 0.5
    min_samples = 5

    # Compute DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(coordinates)
    labels = db.labels_

    # number of clusters and noise points
    no_clusters = len(np.unique(labels))-(1 if -1 in labels else 0)
    no_noise = np.sum(np.array(labels) == -1, axis=0)

    # remove noisy points (points labeled as noise by dbscan), make new array without these
    range_max = len(coordinates)
    filtered_coordinates = np.array([coordinates[i] for i in range(range_max) if labels[i] != -1])
    labels = np.array([labels[i] for i in range(range_max) if labels[i] != -1])

    # sort coordinates into arrays based on which cluster they belong to
    cluster_coords: Dict[int, List[Tuple[float,float]]] = {label: [] for label in labels}
    for i, label in enumerate(labels):
        cluster_coords[label].append(filtered_coordinates[i]) 

    # Average cluster coordinates to find average centres
    # And Find centres coordinate of cluster with most points in it (aware this isnt the most elegant way)
    cluster_centres = []
    longest_index = 0
    length = []
    for cluster_number in cluster_coords:
        length.append(len(cluster_coords[cluster_number]))
        if length[cluster_number] == max(length):
            longest_index = cluster_number
        cluster_centres.append(np.average(cluster_coords[cluster_number], axis=0))

    if debug:
        # Plot points, with cluster with max number of points coloured red and other clusters blue
        colors = list(map(lambda x: '#b40426' if x == longest_index else '#3b4cc0', labels))
        plt.scatter(filtered_coordinates[:,0], filtered_coordinates[:,1], c=colors, marker="x", picker=True)
        plt.title('Clustered data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    return cluster_centres[longest_index]
