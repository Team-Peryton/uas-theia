from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from numpy.lib.npyio import load

from theia.utils import resizeWithAspectRatio, display, drawContours, timer, logger

MIN_AREA = 3000
EPSILON_MULTIPLY = 0.01
SCALE_FACTOR = 5

#cv2.setNumThreads(1)


class Square:
    """ store square attributes """
    x1,x2,x3,x4 = 0,0,0,0 
    y1,y2,y3,y4 = 0,0,0,0
    len12, len23, len34, len41 = 0,0,0,0
    lengths: List[int] = []


def order_points(pts):
	""" https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/  """
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def four_point_transform(image, pts):
	""" 
	https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/ 
	Usage: new_img = four_point_transform(img, squareCountours.reshape(4,2)) 
	"""
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], 
		dtype = "float32"
	)
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped


def calcDistance(x1, y1, x2, y2):
    param1 = (x2 - x1)
    param2 = (y2 - y1)
    inside = (param1 * param1) + (param2 * param2)
    return math.sqrt(inside)


def cropMinAreaRect(img, rect):  # idk how this works but it does
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    imgRot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = imgRot[
        pts[1][1]:pts[0][1],
        pts[1][0]:pts[2][0]
    ]

    return img_crop


def approxContour(contour):
    # fit contour to a simpler shape
    # accuracy is based on EPSILON_MULTIPLY
    epsilon = EPSILON_MULTIPLY * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx


def getSquareLengths(approx: List[List[List[int]]]) -> Square:
    """ 
    UNUSED
    extract lengths from the co-ordinates 
    """
    s = Square()
    
    s.x1, s.y1 = approx[0][0]
    s.x2, s.y2 = approx[1][0]
    s.x3, s.y3 = approx[2][0]
    s.x4, s.y4 = approx[3][0]

    s.len12 = calcDistance(s.x1, s.y1, s.x2, s.y2)
    s.len23 = calcDistance(s.x2, s.y2, s.x3, s.y3)
    s.len34 = calcDistance(s.x3, s.y3, s.x4, s.y4)
    s.len41 = calcDistance(s.x4, s.y4, s.x1, s.y1)

    s.lengths = [s.len12, s.len23, s.len34, s.len41]
    s.lengths.sort()

    return s


def isSquare(s: Square) -> bool:
    """ UNUSED """
    return ((s.lengths[3] - s.lengths[0]) / s.lengths[3]) < 0.1


def filterContours(contours):
    """ """
    squareIndexes = []

    # filter contours
    for i, contour in enumerate(contours):  # for each of the found contours
        if cv2.contourArea(contour) > MIN_AREA:
            approx = approxContour(contour)
            if len(approx) == 4:
                #s = getSquareLengths(approx)
                squareIndexes.append(i)
    
    return squareIndexes


def filterImage(image: np.ndarray) -> np.ndarray:
    """
    LEGACY
    given an image, return an image which shows only white 
    Uses K means
    """
    # downsample, and convert
    height, width = image.shape[:2]
    img = cv2.resize(image, (round(width / SCALE_FACTOR), round(height / SCALE_FACTOR)), interpolation=cv2.INTER_AREA)
    img_blurred = cv2.GaussianBlur(img, (5,5), 0)
    Z = img_blurred.reshape((-1,3))
    Z = np.float32(Z)

    # it's possible this criteria could be optimised
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    _ret, label, center=cv2.kmeans(Z,K,None,criteria,1,cv2.KMEANS_RANDOM_CENTERS)

    # convert back into uint8 and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # convert to black and white
    imgGray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    _thresh, imgf = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return imgf


def target_centre(contour: list) -> Tuple[int, int]:
    """ given the square corners, return the centre of the square """
    x = sum([item[0] for item in contour])/4
    y = sum([item[1] for item in contour])/4
    return int(x), int(y)


def find_targets(image: np.ndarray, debug=False) -> List[List[np.ndarray, Tuple[int,int]]]:
    """ 
    return a colour cropped image of the target, and the centre position within the image
    """
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)
    img_thresh = cv2.adaptiveThreshold(
        imgBlurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY,
        199,
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

    results = []
    for index in squareIndexes:
        hier = hierarchy[0][index]
        
        if hier[3] in squareIndexes:  # if a square has a parent that is also a square
            target_contour = approxContour(contours[index])

            reshaped = target_contour.reshape(4,2)

            centre = target_centre(reshaped)

            # deal with the perspective distortion
            cropped = four_point_transform(image, reshaped)
            
            # TODO shave 10 pixels off from all edges to remove the frame..
            # Shave 10% off instead?
            cropped_further = cropped[10:-10, 10:-10]
            
            if debug:
                display(cropped_further)

            results.append(
                cropped_further,
                centre
            )
        
        else:
            logger.info("square found with no interior square")
            #get the position of the square
            target_contour = approxContour(contours[index])
            
            reshaped = target_contour.reshape(4,2)
            # get the centre of the target, then position
            centre = target_centre(reshaped)

            cropped = four_point_transform(image, reshaped)

            results.append(
                cropped,
                centre
            )
            pass


    if len(results) == 0:
        logger.info("nothing found")

    return results

