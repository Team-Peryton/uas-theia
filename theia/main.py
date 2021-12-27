from typing import List, Tuple
import numpy as np
from theia.spec import ImageRecognitionResult, LocationInfo
from theia.utils import logger


def image_recognition(image: np.ndarray, location_info: LocationInfo) -> List[ImageRecognitionResult]:
    """
    The resulting function of all this mess
    """
    logger.info("got image")
    result = ImageRecognitionResult()
    return result
