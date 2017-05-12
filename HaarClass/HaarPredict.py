import sys
import os
import json
import numpy as np
import cv2
import pickle
import HaarPredictProc as HPP
#####################################################
# Script used for running the Haar Prediction on the Images
#####################################################
###############Delete the results from last prediction run
if "foundBoxed" in os.listdir("."):
	os.system("rm -r foundBoxed")
	os.system("mkdir foundBoxed")
else:
	os.system("mkdir foundBoxed")


sealion_AMcascade = cv2.CascadeClassifier('./AMcascade/cascade.xml')
classifiers = [sealion_AMcascade]

img = cv2.imread("../Train/640.jpg")
predictorClass = HPP.HaarPredictProc(img, classifiers, 512)
img = predictorClass.run((50,50), (90,90), 10) 
cv2.imwrite("./foundBoxed/640.jpg", img)


img = cv2.imread("../Train/100.jpg")
predictorClass = HPP.HaarPredictProc(img, classifiers, 512)
img = predictorClass.run((50,50), (90,90), 10) 
cv2.imwrite("./foundBoxed/100.jpg", img)