import math
import numpy as np
from typing import Tuple
from theia.spec import LocationInfo

RESOLUTION = np.array([4056, 3040]) # px
FOV = math.radians(63)              # FOV : float, Horizontal field of view in degrees
EARTH_RADIUS = 6378137.0            # Radius of "spherical" earth


def triangulate(
        target_centre: Tuple[float, float], location_info: LocationInfo) -> Tuple[float, float]:
    """
    Returns GPS coordinates of target (lat, lon)

    target_centre
        (x,y) pixel position of centre of target
    location_info
        see spec.py
    """

    # convert to useful formats
    uv = np.array(target_centre)            # Target pixel location (px) 
    heading = math.radians(location_info.heading)         # heading (rad)
    pitch = math.radians(location_info.pitch)             # Pitch (rad)
    roll = math.radians(location_info.roll)               # Roll (rad)

    # heading rotation matrix
    R = np.array([
        [math.cos(heading), -math.sin(heading)], 
        [math.sin(heading), math.cos(heading)]
    ])
    
    # Uses trig to calculate the displacement from aircraft to target. 
    # Assumes no distortion and that FOV is proportional to resolution. 
    # X followed by Y. Roll affects X, Pitch affects Y
    delta = np.array([
        location_info.alt * math.tan(roll) + 2 * ((uv[0]/RESOLUTION[0]) - 0.5) * location_info.alt * math.tan(FOV/2 + abs(roll)), 
        location_info.alt * math.tan(pitch) + 2 * ((uv[1]/RESOLUTION[1]) - 0.5) * location_info.alt * math.tan((RESOLUTION[1]/RESOLUTION[0]) * (FOV/2) + abs(pitch))
    ])
    
    # Aligns delta to aircraft heading
    delta = np.matmul(delta, R)
    
    #Coordinate offsets in radians
    d_lat = delta[1]/EARTH_RADIUS
    d_lon = delta[0]/(EARTH_RADIUS*math.cos(math.pi*location_info.lon/180))

    lat = location_info.lat + (d_lat * 180/math.pi)
    lon = location_info.lon + (d_lon * 180/math.pi) 

    return (lat, lon)

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from numpy import array,reshape,hstack,size

def clustering(X,y):
    #dbscan configuration
    epsilon = 0.5
    min_samples = 5

    # Compute DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
    labels = db.labels_
   
    # number of clusters and noise points
    no_clusters = len(np.unique(labels))-(1 if -1 in labels else 0)
    no_noise = np.sum(np.array(labels) == -1, axis=0)

    # remove noisy points (points labeled as noise by dbscan), make new array without these
    range_max = len(X)
    X = np.array([X[i] for i in range(range_max) if labels[i] != -1])
    labels = np.array([labels[i] for i in range(range_max) if labels[i] != -1])

    # sort coordinates into arrays based on which cluster they belong to
    cluster_coords = []
    for i in range(len(np.unique(labels))):
        cluster_coords.append([])
    for i in range(0,len(labels)):
        cluster_coords[labels[i]].append(X[i])
    
    # Average cluster coordinates to find average centres
    # And Find centres coordinate of cluster with most points in it (aware this isnt the most elegant way)
    cluster_centres = []
    longest_index = 0
    length = []
    for i in range(len(np.unique(labels))):
        length.append(len(cluster_coords[i]))
        if length[i]==max(length):
            longest_index = i
        cluster_centres.append(np.average(cluster_coords[i],axis=0))
    
    return(cluster_centres[longest_index],no_clusters,no_noise)
