import cv2
from typing import Tuple
import numpy as np


def calculate_colour(image) -> Tuple[int,int,int]:
    """ given a target, find the target colour """
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    raw_colour = dominant.tolist()
    return int(raw_colour[0]), int(raw_colour[1]), int(raw_colour[2])