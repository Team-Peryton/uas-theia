from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ImageRecognitionResult:
    image_name: str = "" # file name
    charachter: str = ""
    colour: Tuple[int,int,int] = (0, 0, 0) # [R, G, B]
    centre: Tuple[int,int] = (0, 0)
    position: Tuple[float,float] = (0.0, 0.0) # lat, lon
    cropped: np.ndarray = np.array([]) # the cropped image
    no_nested_square: bool = False
