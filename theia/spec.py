from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ImageRecognitionResult:
    image_name: str = "" # file name
    centre: Tuple[int,int] = (0, 0)
    position: Tuple[float,float] = (0.0, 0.0) # lat, lon


@dataclass
class LocationInfo:
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    heading: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
