import math
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from theia.spec import LocationInfo

RESOLUTION = np.array([1080, 1920]) # px
FOV = math.radians(63)              # FOV : float, Horizontal field of view in degrees
EARTH_RADIUS = 6378137.0            # Radius of "spherical" earth


def triangulate(target_centre: Tuple[float, float], location_info: LocationInfo) -> Tuple[float, float]:
    """
    Returns GPS coordinates of target (lat, lon)

    target_centre
        (coordinates,y) pixel position of centre of target
    location_info
        see spec.py
    """

    # convert to useful formats
    # Target pixel location (px)
    uv = np.array(target_centre)
    heading = math.radians(location_info.heading)         # heading (rad)
    pitch = location_info.pitch             # Pitch (rad)
    roll = location_info.roll               # Roll (rad)

    # heading rotation matrix
    R = np.array([
        [math.cos(heading), -math.sin(heading)],
        [math.sin(heading), math.cos(heading)]
    ])

    # Uses trig to calculate the displacement from aircraft to target.
    # Assumes no distortion and that FOV is proportional to resolution.
    # X followed by Y. Roll affects X, Pitch affects Y
    delta = np.array([
        location_info.alt * math.tan(roll) + 2 * (
            (uv[0]/RESOLUTION[0]) - 0.5) * location_info.alt * math.tan(FOV/2 + abs(roll)),
        location_info.alt * math.tan(pitch) + 2 * ((uv[1]/RESOLUTION[1]) - 0.5) * location_info.alt * math.tan(
            (RESOLUTION[1]/RESOLUTION[0]) * (FOV/2) + abs(pitch))
    ])

    # Aligns delta to aircraft heading
    delta = np.matmul(delta, R)

    # Coordinate offsets in radians
    d_lat = delta[1]/EARTH_RADIUS
    d_lon = delta[0]/(EARTH_RADIUS*math.cos(math.pi*location_info.lon/180))

    lat = location_info.lat + (d_lat * 180/math.pi)
    lon = location_info.lon + (d_lon * 180/math.pi)

    return (lat, lon)


def clustering(coordinates: List[Tuple[float, float]], epsilon=0.5, min_samples=5):
    """
    given a list of GPS co-ordinates, find the most likely location of the target.
    epsilon and min_samples as arguements to DBSCAN
    """

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

    return(cluster_centres[longest_index], no_clusters, no_noise)
