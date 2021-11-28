from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ImageRecognitionResult:
    image_name: str = "" # file name
    centre: Tuple[int,int] = (0, 0)
    position: Tuple[float,float] = (0.0, 0.0) # lat, lon
