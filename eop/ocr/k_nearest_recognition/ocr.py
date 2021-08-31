import cv2
import numpy as np
import sys

RESIZED_WIDTH = 30
RESIZED_HEIGHT = 30
MIN_AREA_OCR = 100


class DataContour:
    """ """
    intX = 0  # bounding rect top left corner x location
    intY = 0  # bounding rect top left corner y location
    intWidth = 0  # bounding rect width
    intHeight = 0  # bounding rect height
    floatArea = 0.0  # area of contour

    def checkIfContourIsValid(self):
        """ should have more conditions """
        if self.floatArea < MIN_AREA_OCR:
            return False
        return True


class K_nearest_recognition:
    def __init__(self):
        self.k_nearest = self.load_model()

    def ocr_k_nearest(self, img) -> str:
        """ 
        LEGACY
        given a cropped binary image of a charachter, return the charachter 
        """
        npaContours, _npaHierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        if len(npaContours) > 1:
            raise Exception("should only have one contour here")

        dataContour = self.getRectangle(npaContours[0])
        resized = self.resize(dataContour, img)

        retval, npaResults, neigh_resp, dists = self.k_nearest.findNearest(resized, 3)
        return str(chr(int(npaResults[0][0])))


    def resize(self, dataContour, img_thresh):
        """ convert from cropped orginal resolution to low res """
        # converts to from minimum bounding square to minimum bounding rectangle
        imgROI = img_thresh[
                dataContour.intY: dataContour.intY + dataContour.intHeight,
                dataContour.intX: dataContour.intX + dataContour.intWidth
                ]
        imgROIResized = cv2.resize(imgROI, (RESIZED_WIDTH, RESIZED_HEIGHT))
        npaROIResized = np.float32(imgROIResized.reshape((1, RESIZED_WIDTH * RESIZED_HEIGHT)))
        return npaROIResized


    def getRectangle(self, npaContour) -> DataContour:
        """ given contours, calculate rectangle information """
        dataContour = DataContour()
        dataContour.intX, dataContour.intY, dataContour.intWidth, dataContour.intHeight = cv2.boundingRect(npaContour)
        dataContour.floatArea = cv2.contourArea(npaContour)
        if dataContour.checkIfContourIsValid():
            return dataContour
        else:
            raise Exception("contour invalid")


    def load_model(self):
        """ load data sources and train model """
        try:
            npaClassifications = np.loadtxt('nyx/k_nearest_recognition/classifications.txt', np.float32)
            npaFlattenedImages = np.loadtxt('nyx/k_nearest_recognition/imageData.txt', np.float32)
        except:
            sys.exit('text files not present')

        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
        k_nearest = cv2.ml.KNearest_create()
        k_nearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        return k_nearest
