dataset = r"C:\Users\olive\Documents\GitHub\image-recognition\dataset\letter_set"
import cv2

from nyx.recognition import target_recognition
import os
from nyx.utils import display


def ocr():
    for folder in os.listdir(dataset):
        print(folder)
        for letter in os.listdir(os.path.join(dataset,folder)):
            img = cv2.imread(os.path.join(dataset,folder,letter))
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0) 
            imgThresh = cv2.adaptiveThreshold(imgBlurred, 255,  #pass threshold = white
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, #invert, background=black
                                        11, 2)
            try:
                letter = target_recognition.ocr(imgThresh)
                print(letter==folder)
            except:
                print("contour invalid")
