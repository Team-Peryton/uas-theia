from typing import List, Tuple

import cv2
import numpy as np

from theia.utils import display, logger


def approxContour(contour: list, options: dict):
    """
    fit contour to a simpler shape
    accuracy is based on EPSILON_MULTIPLY
    """
    epsilon = options['epsilon'] * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def solidity(approx: list) -> float:
    """
    Compares the actual area with the convex hull area
    """
    area = cv2.contourArea(approx)
    hull = cv2.convexHull(approx)
    hull_area = cv2.contourArea(hull)
    return float(area)/hull_area


def filterContours(contours: list, options: dict):
    """ 
    return only the contours that are squares
    """
    squareIndexes = []
    for i, contour in enumerate(contours):  # for each of the found contours
        if cv2.contourArea(contour) > options["min_area"]: # remove any tiny noise bits
            approx = approxContour(contour, options)
            if len(approx) in options["sides"]:
                if solidity(approx) > options["min_solidity"]:
                    squareIndexes.append(i)

    return squareIndexes


def square_target_centre(contour: list) -> Tuple[int, int]:
    """ 
    given the square corners, return the centre of the square 
    """
    x = sum([item[0] for item in contour])/4
    y = sum([item[1] for item in contour])/4
    return int(x), int(y)


def find_targets(image: np.ndarray, options) -> List[Tuple[int,int]]:
    """ 
    return the squaree centre position within the image, in pixels
    """
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (options["ksize"], options["ksize"]), options["sigma"])
    img_thresh = cv2.adaptiveThreshold(
        imgBlurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY,
        options["block_size"],
        options["c"]
    )
    
    if options["debug"]: 
        display(img_thresh)

    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    squareIndexes = filterContours(contours, options)
    
    if options["debug"]:
        for index in squareIndexes:
            cv2.drawContours(image, contours[index], -1, (0, 255, 0), 5)
        display(image)

    # this for loop is mainly to check if there are multiple squares in the same image
    # otherwise there would not be a loop
    results = []
    for index in squareIndexes:
        # this step has already been done, so potentially filterContours should return the target_contour instead
        target_contour = approxContour(contours[index], options)
        reshaped = target_contour.reshape(4,2)
        centre = square_target_centre(reshaped)
        results.append(centre)

    return results
