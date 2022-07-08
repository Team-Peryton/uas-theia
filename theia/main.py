
from datetime import datetime
from distutils.log import debug
from typing import List
from picamera2.picamera2 import Picamera2, Preview
from dronekit import connect

import cv2
import numpy as np
from legacy.clustering import cluster

from theia.image_segmentation import find_targets
from theia.position_estimation import triangulate, clustering, exclude_outside_perimeter
from theia.spec import ImageRecognitionResult, LocationInfo
from theia.utils import logger


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
        
        logger.warn("Configuring the camera")
        self.picam = Picamera2(verbose_console=0)
        config = self.picam.still_configuration(main={"size": (1920,1080)}) #these 3 lines should turn off preview
        self.picam.configure(config)
        self.picam.start_preview(Preview.NULL)
        self.picam.start()

        logger.warn("Connecting to Ardupilot")
        self.vehicle = connect('/dev/ttyACM0', baud=56700, wait_ready=True)
        
        logger.info(self.picam)
        logger.info(self.vehicle)


    def image_recognition(self, image: np.ndarray, location_info: LocationInfo, name:str) -> None:
        """
        The resulting function of all this mess
        """
        logger.info("processing image")
        target_pixel_locations = find_targets(image, self.options)
        for target in target_pixel_locations:
            location = triangulate(target, location_info)
            result = ImageRecognitionResult(image_name=name, centre=target, position=location)
            logger.info(result)
            self.found_targets.append(result)
    

    def image_recognition_from_files(self):
        start_height = 142
        files = [f for f in os.listdir(f"/media/pi/USB DISK/)"]

        for image_number in range(0,len(files)):
            image = cv.imread(directory + "/" + files[image_number])
            name = files[image_number].split(",")
            l = LocationInfo(alt=name[1], lon=name[2], lat=name[3], heading=name[4], roll=name[5], rollspeed=name[6], pitch=name[7], pitchspeed=name[8], yaw=name[9], yawspeed=name[10])
            self.image_recognition(image,l, name)


    def calaculate_target_position(self):
        """
        from the targets found, estimate the actual centre of the target
        done just before landing
        """
        logger.info("Starting clustering")
        coordinates = [result.position for result in self.found_targets]
        logger.info(f"got raw co-ords{coordinates}")
        #coordinates = exclude_outside_perimeter(coordinates)
        logger.info(f"got filtered co-ords {coordinates}")
        target_location = clustering(coordinates)

        logger.info(f"""
        ------------------------------
        FINAL TARGET POSITION ESTIMATE 
        {target_location[0]} with {target_location[1]} clusters found and {target_location[2]} outliers
        ------------------------------
        """)
        return target_location
    
    
    def get_found_targets(self):
        """ debug tool """
        logger.info("--- Target Position Information: ---")
        logger.info(f"{len(self.found_targets)} images with targets")
        for i, row in enumerate(self.found_targets):
            logger.info(f"{i} -> {row}")
        return self.found_targets
    

    def get_location_info(self) -> LocationInfo:
        location = self.vehicle.location.global_relative_frame
        heading = self.vehicle.heading
        
        roll = self.vehicle._roll
        rollspeed = self.vehicle._rollspeed
        
        pitch = self.vehicle._pitch
        pitchspeed = self.vehicle._pitchspeed
        
        yaw = self.vehicle._yaw
        yawspeed = self.vehicle._yawspeed

        return LocationInfo(location.lat, location.lon, location.alt, 
            heading, 
            pitch, pitchspeed, 
            roll, rollspeed, 
            yaw, yawspeed
        )
    
    def get_image(self) -> np.ndarray:
        return self.picam.capture_array()
