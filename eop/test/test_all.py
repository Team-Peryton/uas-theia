import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def test_one():
    assert 1 == 1


def test_image_recognition_accuracy():
    """ test against the dataset """
    
    from nyx import target_recognition

    files = [f for f in os.listdir('./dataset/sim_dataset/')]

    results = []
    for image_name in files:
        image = cv2.imread('./dataset/sim_dataset/' + image_name)
        result = target_recognition.findCharacters(image)
        print(result)
        results.append(result)
    
    def check(x) -> list:
        try: return x[0]['charachter']
        except: return []

    score = sum([1 if 'T' in check(i) else 0 for i in results])/len(results)

    print(f"{score*100}% dataset accuracy")

    assert score/len(results) > 0.95


def test_image_recognition_speed():
    from nyx import target_recognition

    files = [f for f in os.listdir('./dataset/sim_dataset/')]
    iterations = 100
    img = cv2.imread('./dataset/sim_dataset/' + files[10])
    
    start = time.time()
    for i in range(iterations):
        print(f"found charachters {target_recognition.findCharacters(img)} in image")

    print(f"average time taken per image {(time.time() - start) / (iterations)}")

    assert 1==1


def test_triangulate():
    """ """

    from nyx import target_recognition

    position = (0,0)
    target_px = (1920,1440)
    altitude = 25.0
    heading = 90

    result = target_recognition.triangulate(position, target_px, altitude, heading)
    assert(result == (0.0,0.0))
    
    target_px = (0,1440)
    heading = 0
    altitude = 1
    result = target_recognition.triangulate(position, target_px, altitude, heading)
    result = round(result[0], 3)
    assert(result == -1.192)
