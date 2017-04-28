import numpy as np


class CMG:
    def __init__(self, ColorClassifier):
        self.CC = ColorClassifier.coef_[0]

    def getMaskImg(self, img):
        tempBlue = img[:,:,0]
        tempGreen = img[:,:,1]
        tempRed = img[:,:,2]
        return tempRed
        tempBlue = 1.0 + np.exp(-tempBlue * self.CC[0] - tempGreen * self.CC[1] - tempRed * self.CC[2] - tempBlue * self.CC[3] / tempGreen - tempBlue * self.CC[4] / tempRed - tempGreen * self.CC[5] / tempRed)
        tempBlue =  (tempBlue ** -1)
        return tempBlue.astype('uint8')