import math
import numpy as np
from typing import Tuple

RESOLUTION = np.array([4056, 3040]) # px
FOV = math.radians(63)              # FOV : float, Horizontal field of view in degrees
EARTH_RADIUS = 6378137.0            # Radius of "spherical" earth


def triangulate(
        position: Tuple[float, float], 
        target_centre: Tuple[int, int], 
        altitude: float, 
        heading: float,
        pitch: float,
        roll: float,
    ):
    """
    Returns GPS coordinates of target (lat, lon)

    position : float[]
        (lat,lon) GPS co-ordinates of where the image was taken
    target_uv : int[]
        (x,y) pixel position of centre of target
    altitude : float
        Altitude in metres, relative to ground
    heading : float
        Aircraft heading in degrees
    pitch : float
        Aircraft pitch in degrees
    roll : float
        Aircraft roll in degrees
    """

    # convert to useful formats
    uv = np.array(target_centre)            # Target pixel location (px) 
    heading = math.radians(heading)         # heading (rad)
    pitch = math.radians(pitch)             # Pitch (rad)
    roll = math.radians(roll)               # Roll (rad)

    # heading rotation matrix
    R = np.array([
        [math.cos(heading), -math.sin(heading)], 
        [math.sin(heading), math.cos(heading)]
    ])
    
    # Uses trig to calculate the displacement from aircraft to target. 
    # Assumes no distortion and that FOV is proportional to resolution. 
    # X followed by Y. Roll affects X, Pitch affects Y
    delta = np.array([
        altitude * math.tan(roll) + 2 * ((uv[0]/RESOLUTION[0]) - 0.5) * altitude * math.tan(FOV/2 + abs(roll)), 
        altitude * math.tan(pitch) + 2 * ((uv[1]/RESOLUTION[1]) - 0.5) * altitude * math.tan((RESOLUTION[1]/RESOLUTION[0]) * (FOV/2) + abs(pitch))
    ])
    
    # Aligns delta to aircraft heading
    delta = np.matmul(delta, R)
    
    #Coordinate offsets in radians
    d_lat = delta[1]/EARTH_RADIUS
    d_lon = delta[0]/(EARTH_RADIUS*math.cos(math.pi*position[1]/180))

    lat = position[0] + (d_lat * 180/math.pi)
    lon = position[1] + (d_lon * 180/math.pi) 

    return (lat, lon)
