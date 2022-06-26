from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

from legacy.clustering import cluster


# Configuration options
num_samples_total = 1000

# Remove this when using GPS coordinate inputs
cluster_centers = [(32.30640,32.30650), (21.61450,21.61460),(25.00005,26.0003)]

num_classes = len(cluster_centers)

# Remove this when using GPS coordinate inputs
coordinates, y = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 0.5)

centre = cluster(coordinates)

print(centre)