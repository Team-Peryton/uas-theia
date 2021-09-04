import numpy as np
import cv2
import os

minArea = 100
resizedWidth = 30
resizedHeight = 30

path = './training/' 

def main():
    trainingFiles = [f for f in os.listdir(path) if os.path.splitext(f)[1] == '.png'] # all png files in training folder

    if len(trainingFiles) == 0: #if no files found
        print('no training files found')
        return #exit main

    print('training data found for these characters:')
    print([x[0] for x in trainingFiles])
    print('starting training')
    
    npaFlattenedImages = np.empty((0, resizedWidth * resizedHeight))
    intClassifications = [] #to hold list of character values

    for files in trainingFiles: #for each training picture
        imgTraining = cv2.imread(path + files) #read a char's file
        
        imgGray = cv2.cvtColor(imgTraining, cv2.COLOR_BGR2GRAY) # grayscale image
        imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)        # blur

        imgThresh = cv2.adaptiveThreshold(imgBlurred, 255,  #pass threshold = white
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, #invert, background=black
                                          11, 2)

        imgThreshCopy = imgThresh.copy()    #finding contours modifies image, make copy

        npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for npaContour in npaContours:  #for each contour found in image
            if cv2.contourArea(npaContour) > minArea:  # if contour is big enough to consider
                [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

                imgROI = imgThresh[intY:intY+intH, intX:intX+intW] #cropped character from threshold
                imgROIResized = cv2.resize(imgROI, (resizedWidth, resizedHeight))

                intClassifications.append(ord(files[0])) #add ascii code of char to classification list

                npaFlattenedImage = imgROIResized.reshape((1, resizedWidth * resizedHeight))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

    fltClassifications = np.array(intClassifications, np.float32) #convert array of ints into floats
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    print('training complete, writing files')
    np.savetxt('classifications.txt', npaClassifications)
    np.savetxt('imageData.txt', npaFlattenedImages)
    print('files written, exiting')


if __name__ == '__main__':
    main()
