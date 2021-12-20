from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from theia.utils import resizeWithAspectRatio, display, drawContours, logger

MIN_AREA = 3000
EPSILON_MULTIPLY = 0.01
SCALE_FACTOR = 5


def approxContour(contour):
    """
    fit contour to a simpler shape
    accuracy is based on EPSILON_MULTIPLY
    """
    epsilon = EPSILON_MULTIPLY * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def filterContours(contours):
    """ 
    return only the contours that are squares
    """
    squareIndexes = []

    # filter contours
    for i, contour in enumerate(contours):  # for each of the found contours
        if cv2.contourArea(contour) > MIN_AREA:
            approx = approxContour(contour)
            if len(approx) == 4:
                #s = getSquareLengths(approx)
                squareIndexes.append(i)
    
    return squareIndexes


def target_centre(contour: list) -> Tuple[int, int]:
    """ 
    given the square corners, return the centre of the square 
    """
    x = sum([item[0] for item in contour])/4
    y = sum([item[1] for item in contour])/4
    return int(x), int(y)


def find_targets(image: np.ndarray, debug=False) -> List[List[Tuple[int,int]]]:
    """ 
    return the centre position within the image
    """
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)
    img_thresh = cv2.adaptiveThreshold(
        imgBlurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY,
        220,
        -30
    )
    
    if debug: 
        display(img_thresh)

    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    squareIndexes = filterContours(contours)
    
    if debug:
        for contour in contours:
            cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
        display(image)

    # this for loop is mainly to check if there are multiple squares in the same image
    # otherwise there would not be a loop
    results = []
    for index in squareIndexes:
        target_contour = approxContour(contours[index])

        reshaped = target_contour.reshape(4,2)

        centre = target_centre(reshaped)

        results.append(
            centre
        )


    if len(results) == 0:
        logger.info("nothing found")

    return results


if __name__ == "__main__":
    pass