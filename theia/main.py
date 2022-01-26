from datetime import datetime
from typing import List

import cv2
import numpy as np

from theia.image_segmentation import find_targets
from theia.position_estimation import triangulate
from theia.spec import ImageRecognitionResult, LocationInfo
from theia.utils import logger


class ImageRecognition:

    def __init__(self, file_base_directory):
        self.options = {
            "block_size": 249,
            "c": -39,
            "ksize": 49,
            "sigma": 0,
            "epsilon": 0.02,
            "square_ar": 0.4,
            "min_area": 3000,
        }
        self.file_base_directory = file_base_directory
        self.found_targets: ImageRecognitionResult = []


    def image_recognition(self, image: np.ndarray, location_info: LocationInfo) -> List[ImageRecognitionResult] | None:
        """
        The resulting function of all this mess
        """
        logger.info("processing image")
        target_pixel_locations = find_targets(image, self.options)
        for target in target_pixel_locations:
            location = triangulate(target, location_info)
            image_name = self.file_base_directory + "\\runtime\\" + datetime.now().strftime("%d%m%Y_%H-%M-%S.%f") + ".jpg"
            cv2.imwrite(image_name, image)
            result = ImageRecognitionResult(image_name=image_name, centre=target, position=location)
            logger.info(result)
            self.found_targets.append(result)
        return None


    def calaculate_target_position(self):
        """ 
        from the targets found, estimate the actual centre of the target
        done just before landing
        """
        pass
