import numpy as np
import cv2
import os

resizedWidth = 50
resizedHeight = 50
dataset = r"C:\Users\olive\Documents\GitHub\image-recognition\dataset\letter_set"

npaFlattenedImages = np.empty((0, resizedWidth * resizedHeight))
intClassifications = [] #to hold list of character values


for folder in os.listdir(dataset):
    print(folder)
    for letter in os.listdir(os.path.join(dataset,folder)):
        img = cv2.imread(os.path.join(dataset,folder,letter))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)        # blur
        imgThresh = cv2.adaptiveThreshold(imgBlurred, 255,  #pass threshold = white
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, #invert, background=black
                                    11, 2)
        npaContours, npaHierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(npaContours) == 1:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContours[0])
            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]
            imgROIResized = cv2.resize(imgROI, (resizedWidth, resizedHeight))
            intClassifications.append(ord(folder)) #add ascii code of char to classification list
            npaFlattenedImage = imgROIResized.reshape((1, resizedWidth * resizedHeight))
            npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)


fltClassifications = np.array(intClassifications, np.float32) #convert array of ints into floats
npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))
print('training complete, writing files')
np.savetxt('classifications2.txt', npaClassifications)
np.savetxt('imageData2.txt', npaFlattenedImages)
print('files written, exiting')