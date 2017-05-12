import numpy as np
import cv2
import sys

class CMG:
    def __init__(self, ColorClassifier = None):
        self.ColC = ColorClassifier
        ########## Histogram Equalization
        self.clahe = cv2.createCLAHE()
    
    def getMaskImg(self, img):
        #### WAs used when I was playing around with stuff Dead Code
        if self.ColC is None:
            print("Mask generator does not have a color classifier")
            return img
        else:
            self.CC = self.ColC._coef
        tempBlue = img[:,:,0]
        tempGreen = img[:,:,1]
        tempRed = img[:,:,2]
        result = tempRed - tempGreen
         
        return tempRed - tempBlue

        tempBlue[tempBlue < 1.0] = 1.0
        tempGreen[tempGreen < 1.0] = 1.0
        tempRed[tempRed < 1.0] = 1.0
        result = 1.0 + np.exp(-tempBlue * self.CC[0] - tempGreen * self.CC[1] - tempRed * self.CC[2] - tempBlue * self.CC[3] / tempGreen - tempBlue * self.CC[4] / tempRed - tempGreen * self.CC[5] / tempRed)
        result =  (tempBlue ** -1)
        tempBlue[result > .5] = 0
        tempGreen[result > .5] = 0
        tempRed[result > .5] = 0
        #tempBlue[result < .5] = 50
        #tempRed[result < .5] = 50
        #tempGreen[result < .5] = 50
        img = cv2.cvtColor(np.dstack((tempBlue, tempGreen, tempRed)), cv2.COLOR_BGR2GRAY)

        return img

    def getPlayImg(self, img):
        return self.clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        
    def getInitialMask(self, img):
        #####Function used for masking the images for the Haar Classifier
        img[img < 40] = 0
        img[img[:,:, 2] < 80] = 0
        img[img < 5] = 5
        img[:,:][img[:,:,0] - 5 > img[:,:,2]] = 0
        img[:,:][img[:,:,1] - 5 > img[:,:,2]] = 0
        imgBlurr = cv2.GaussianBlur(img, (5,5), 1)
        edges = cv2.absdiff(img, imgBlurr)
        img = cv2.absdiff(edges, img)
        img[img < 50] = 0
       
    
        return img