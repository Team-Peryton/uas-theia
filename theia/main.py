from typing import List
import numpy as np
from theia.spec import ImageRecognitionResult

def image_recognition(image: np.ndarray) -> List[ImageRecognitionResult]:
    """
    The resulting function of all this mess
    """
    return ImageRecognitionResult