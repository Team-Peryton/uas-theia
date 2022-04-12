from datetime import datetime
from distutils.log import debug
from typing import List

import cv2
import numpy as np

from theia.image_segmentation import find_targets
from theia.position_estimation import triangulate
from theia.spec import ImageRecognitionResult, LocationInfo
from theia.utils import logger
from theia.clustering import cluster


class ImageRecognition:
    """
    all image rec functionality is here
    """

    def __init__(self, file_base_directory):
        self.options = {
            "block_size": 399,
            "c": -39,
            "ksize": 39,
            "sigma": 0,
            "epsilon": 0.1,
            "min_area": 1000,
            "sides":[4],
            "min_solidity": 0.6,
            "debug": False
        }
        self.file_base_directory = file_base_directory
        self.found_targets: List[ImageRecognitionResult] = []


    def image_recognition(self, image: np.ndarray, location_info: LocationInfo) -> None:
        """
        The resulting function of all this mess
        """
        logger.info("processing image")
        target_pixel_locations = find_targets(image, self.options)
        for target in target_pixel_locations:
            location = triangulate(target, location_info)
            image_name = f"{self.file_base_directory} \\runtime\\ {datetime.now().strftime('%d%m%Y_%H-%M-%S.%f')}.jpg"
            cv2.imwrite(image_name, image)
            result = ImageRecognitionResult(image_name=image_name, centre=target, position=location)
            logger.info(result)
            self.found_targets.append(result)


    def calaculate_target_position(self):
        """
        from the targets found, estimate the actual centre of the target
        done just before landing
        """
        logger.info("Starting clustering")
        coordinates = [result.position for result in self.found_targets]
        target_location = cluster(coordinates)
        return target_location
    
    
    def get_found_targets(self):
        """ debug tool """
        logger.info(self.found_targets)
        return self.found_targets