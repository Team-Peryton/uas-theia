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
