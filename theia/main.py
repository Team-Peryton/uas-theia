
import os
from typing import List, Tuple

import cv2
import numpy as np
from dronekit import connect
from picamera2.picamera2 import Picamera2, Preview

from theia.image_segmentation import find_targets
from theia.position_estimation import (clustering, exclude_outside_perimeter,
                                       triangulate)
from theia.spec import ImageRecognitionResult, LocationInfo
from theia.utils import logger


class ImageRecognition:
    """
    all image rec functionality is summed up here.
    Note return types and class variables
    """

    def __init__(self, file_base_directory:str):
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
        } # options used for square recognition

        self.file_base_directory = file_base_directory # used for image storage location
        self.found_targets: List[ImageRecognitionResult] = [] # any image rec result is stored here for later reference
        
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
        Live processes an image and location to predict target location
        """
        logger.info("processing image")
        target_pixel_locations = find_targets(image, self.options)
        for target in target_pixel_locations:
            location = triangulate(target, location_info)
            result = ImageRecognitionResult(image_name=name, centre=target, position=location)
            logger.info(result)
            self.found_targets.append(result)
    

    def image_recognition_from_files(self, start_height:float) -> None:
        """ 
        Post processess images written to disk
        UNTESTED
        """
        files:List[str] = [f for f in os.listdir(f"/media/pi/USB DISK/")]

        for image_number in range(0,len(files)):
            image:np.ndarray = cv2.imread(self.file_base_directory + "/" + files[image_number])
            name:List[str] = files[image_number].split(",")
            location = LocationInfo(alt=float(name[1]) - start_height, 
                lon=float(name[2]), 
                lat=float(name[3]), 
                heading=float(name[4]), 
                roll=float(name[5]), 
                rollspeed=float(name[6]), 
                pitch=float(name[7]), 
                pitchspeed=float(name[8]), 
                yaw=float(name[9]), 
                yawspeed=float(name[10])
            )
            self.image_recognition(image, location, files[image_number])


    def calaculate_target_position(self) -> Tuple[float, float]:
        """
        from the targets found, estimate the actual centre of the target
        done just before landing
        """
        logger.info("Starting clustering")
        coordinates = [result.position for result in self.found_targets]
        logger.info(f"got raw co-ords{coordinates}")
        coordinates = exclude_outside_perimeter(coordinates)
        logger.info(f"got filtered co-ords {coordinates}")
        target_location = clustering(coordinates)

        logger.info(f"""
        ------------------------------
        FINAL TARGET POSITION ESTIMATE 
        {target_location[0]} with {target_location[1]} clusters found and {target_location[2]} outliers
        ------------------------------
        """)
        return target_location[0]
    
    
    def get_found_targets(self) -> List[ImageRecognitionResult]:
        """ 
        debug tool 
        """
        logger.info("--- Target Position Information: ---")
        logger.info(f"{len(self.found_targets)} images with targets")
        for i, row in enumerate(self.found_targets):
            logger.info(f"{i} -> {row}")
        return self.found_targets
    

    def get_location_info(self) -> LocationInfo:
        """ 
        package up Ardupilot position stuff 
        """
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
        """ 
        get an image from the camera as an array 
        """
        return self.picam.capture_array()
