import math
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from theia.spec import LocationInfo

from shapely import geometry

RESOLUTION = np.array([1080, 1920]) # px
ASPECT_RATIO = RESOLUTION[1]/RESOLUTION[0]
FOV = math.radians(63)
EARTH_RADIUS = 6378137.0            # Radius of "spherical" earth

# from https://stackoverflow.com/questions/31257664/relation-between-horizontal-vertical-and-diagonal-field-of-view
SENSOR_W = 6.4 #mm
SENSOR_H = SENSOR_W / ASPECT_RATIO #mm
LENS_F = 6 #mm
FOV_X = 2 * math.atan2(SENSOR_W/2, LENS_F)
FOV_Y = 2 * math.atan2(SENSOR_H/2, LENS_F)


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

    # heading rotation matrix
    R = np.array([
        [math.cos(heading), -math.sin(heading)],
        [math.sin(heading), math.cos(heading)]
    ])

    # Uses trig to calculate the displacement from aircraft to target.
    # Assumes no distortion and that FOV is proportional to resolution.
    # X followed by Y. Roll affects X, Pitch affects Y
    delta = np.array([
        # X
        (
            location_info.alt * math.tan(location_info.roll)
        ) + (
            2 * ((uv[0]/RESOLUTION[0]) - 0.5)
        ) * (
            location_info.alt * math.tan(FOV_X/2 + abs(location_info.roll))
        ),

        # Y
        (
            location_info.alt * math.tan(location_info.pitch)
        ) + (
            2 * (((uv[1]/RESOLUTION[1])) - 0.5)
        ) * (
            location_info.alt * math.tan((RESOLUTION[1]/RESOLUTION[0]) * (FOV_Y/2) + abs(location_info.pitch))
        )
    ])

    # Aligns delta to aircraft heading
    delta = np.matmul(delta, R)

    # Coordinate offsets in radians
    # this is using a simple model so could be improved
    d_lat = delta[1]/EARTH_RADIUS
    d_lon = delta[0]/(EARTH_RADIUS*math.cos(math.pi*location_info.lon/180))

    lat = location_info.lat + (d_lat * 180/math.pi)
    lon = location_info.lon + (d_lon * 180/math.pi)

    return (lat, lon)


def exclude_outside_perimeter(coordinates: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    # takes all cluster centres as inputs and excludes anything outside the competition perimeter
    # input competition perimeter GPS coordinates prior to competition. 
    
    # GPS coords of airfield
    perimeter = [(52.779413, -0.704785),
(52.781214, -0.702219),
(52.783095, -0.706073),
(52.784349, -0.707847),
(52.785987, -0.711378),
(52.785614, -0.713564),
(52.782835, -0.715637),
(52.778343, -0.712970)]

    line = geometry.LineString(perimeter)
    perimeter_poly = geometry.Polygon(line)
    target_centres = []
    
    # probably a better way to do it than the loop
    for i in range(len(coordinates)):
        point_X = coordinates[i][0]
        point_Y = coordinates[i][1]
        point = geometry.Point(point_X, point_Y)
        if perimeter_poly.contains(point):
            target_centres.append(point)

    return(target_centres)


def clustering(coordinates: List[Tuple[float, float]], epsilon=0.0006, min_samples=5):
    """
    given a list of GPS co-ordinates, find the most likely location of the target.
    epsilon and min_samples as arguments to DBSCAN
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



