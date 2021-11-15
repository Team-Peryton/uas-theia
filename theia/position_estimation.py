import math
import numpy as np
from typing import Tuple

RESOLUTION = np.array([3840, 2880]) # px
FOV = math.radians(100)

def triangulate(position_input: Tuple[float, float], target_uv: Tuple[int, int], altitude: float, heading: float):
    """
    Returns absolute coordinates of target in metres

    camera : float[]
        (x,y) GPS location of the camera in metres
    target_uv : float[]
        (x,y) pixel position of centre of target
    altitude : float
        Altitude in metres
    FOV : float
        Horizontal field of view in degrees
    heading : float
        Aircraft bearing in degrees
    """
    position = np.array(position_input)     # Camera GPS (m)
    uv = np.array(target_uv)                # Target pixel location (px) 
    bearing = math.radians(heading)         # Bearing (rad)
    h = altitude                            # Altitude (m)

    # Bearing rotation matrix
    R = np.array([
        [math.cos(bearing), -math.sin(bearing)], 
        [math.sin(bearing), math.cos(bearing)]
        ])
    # Uses trig to calculate the displacement from aircraft to target. 
    # Assumes no distortion and that FOV is proportional to resolution. 
    Delta = np.array([2 * ((uv[0]/RESOLUTION[0]) - 0.5) * h * math.tan(FOV/2), 2 * ((uv[1]/RESOLUTION[1]) - 0.5) * h * math.tan((RESOLUTION[1]/RESOLUTION[0]) * FOV/2)])
    # Aligns Delta to aircraft bearing. 
    Delta = np.matmul(Delta, R)
    
    # Adds Delta to current aircraft location. 
    target_gps = position + Delta

    return tuple(target_gps)