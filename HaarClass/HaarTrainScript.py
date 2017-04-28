import os
import json
import cv2
import numpy as np
import time
import sys
import pickle

sys.path.append("../")
from GenTrainImages import ColorMaskGenerator
IMAGEPART = 256
fp = open("../SeaLionCC.pickle", 'r')
CC = pickle.load(fp)
fp.close()

CMG = ColorMaskGenerator.CMG(CC)

background = open("bg.txt", 'w')
positive = open("sealions.txt" , 'w')

nPOSSamples = 0
nNEGSamples = 0

if 'negimg' in os.listdir("."):
    os.system("rm -r negimg")

os.system("mkdir negimg")

if 'posimg' in os.listdir("."):
    os.system("rm -r posimg")

os.system("mkdir posimg")

fp = open("../SeaLionLoc.json", 'r')
data = json.load(fp)
fp.close()

fp = open("../SeaLionB.json", 'r')
dataB = json.load(fp)
fp.close()


def getBBCrop(BB, subI):
    if BB[1][0] < subI[0][0]:
        Xmin = 0
    else:
        Xmin = BB[1][0] - subI[0][0] - 1
    
    if BB[1][1] < subI[1][0]:
        Ymin = 0
    else:
        Ymin = BB[1][1] - subI[1][0] - 1
    
    
    if BB[0][0] > subI[0][1]:
        Xmax = IMAGEPART - 1 - Xmin
    else:
        Xmax = (BB[0][0] - subI[0][0]) - Xmin
    
    if BB[0][1] > subI[1][1]:
        Ymax = IMAGEPART - 1 - Ymin
    else:
	Ymax = (BB[0][1] - subI[1][0]) - Ymin
    
    return [(Xmin, Ymin), (Xmax, Ymax)]


def spotsInSub(name, subImg):
    result = []
    for key in data[name].keys():
        if key != 'pups' and key != 'error':
            for spot in data[name][key].keys():
                nextS = data[name][key][spot]
                ####Check to see if the spot is inside our subImg
                if nextS[0] > subImg[0][0] and nextS[0] < subImg[0][1] and nextS[1] > subImg[1][0] and nextS[1] < subImg[1][1]:
                    result.append(getBBCrop(dataB[name][key][spot], subImg))
    if result:
        return result
    else:
        return None
                      

def genEvenPartitions(name, img):
    global nNEGSamples, nPOSSamples
    imgShape = np.shape(img)
    print imgShape
    for i in xrange(1, imgShape[0] // IMAGEPART, 1):
        Xshift = i * IMAGEPART
        for j in xrange(1, imgShape[1] // IMAGEPART, 1):
            Yshift = j * IMAGEPART
            spots = spotsInSub(name, [(Xshift - IMAGEPART, Xshift),(Yshift - IMAGEPART, Yshift)])
            subImg =  img[Xshift - IMAGEPART:Xshift, Yshift - IMAGEPART:Yshift].copy()
            #subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY)
            subImg = CMG.getMaskImg(subImg)
            if spots is None:
                cv2.imwrite("./negimg/" + name.split(".")[0] + str(i) + str(j) + ".jpg", subImg)
                background.write("negimg/" + name.split(".")[0] + str(i) + str(j) +".jpg")
                nNEGSamples += 1
            else:

                cv2.imwrite("./posimg/" + name.split(".")[0] + str(i) + str(j) + ".jpg", subImg)
                positive.write("posimg/" + name.split(".")[0] + str(i) + str(j) + ".jpg  " + str(len(spots)) + " ")
                for spot in spots:
                    positive.write(" " + str(spot[0][1]) + " " + str(spot[0][0]) + " " + str(spot[1][1]) + " " + str(spot[1][0]))
                    #cv2.rectangle(subImg, (spot[0][1] + spot[1][1], spot[0][0] + spot[1][0]), (spot[0][1], spot[0][0]), 255)
                    nPOSSamples += 1
                #cv2.imwrite("../TrainBoxed/" + name.split(".")[0] + str(i) + str(j) + ".jpg", subImg)
            if spots is None:
                background.write("\n")
            else:
                positive.write("\n")

for key in data.keys():

	if key != ".gitignore":
	    img = cv2.imread("../Train/" + key)
	    genEvenPartitions(key, img)

#img = cv2.imread("../Train/44.jpg")
#genEvenPartitions("44.jpg", img)


background.close()
background = open("bg.txt", 'r')
finalBack = background.read()[:-1]
background.close()
background = open("bg.txt", 'w')
background.write(finalBack)
background.close()

positive.close()
positive = open("sealions.txt", 'r')
finalPos = positive.read()[:-1]
positive.close()
positive = open("sealions.txt", 'w')
positive.write(finalPos)
positive.close()

os.system("opencv_createsamples -vec positive_samples.vec -bg bg.txt -info sealions.txt -bgthresh 80 -num " + str(nPOSSamples) + " -h 50 -w 50")
#os.system("opencv_traincascade -data . -vec positive_samples.vec -bg bg.txt -numPos " + str(nPOSSamples // 40) + " -numNeg " + str(nNEGSamples // 40) + " -h 24 -w 24")   #-minHitRate .9