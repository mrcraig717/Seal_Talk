import numpy as np


class CMG:
    def __init__(self, ColorClassifier):
        self.CC = ColorClassifier.coef_[0]

    def getMaskImg(self, img):
        print("AAAAAAAAAAAAAAAAAAAAAA")
        print(self.CC)
        #img = img.astype(float)
        #img[img < 1.0] = 1.0
        #tempBlue = img[:][:][0]
        tempGreen = img[:][:][1]
        tempRed = img[:][:][2]
        #img[:][:] = 1.0 + np.exp(tempBlue * self.CC[0])
        #img[:][:] = img[:][:] ** -1
        return img.astype('uint8')