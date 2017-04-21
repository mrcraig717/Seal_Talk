from GenJSON import GenJSON
from ColorClassifier import ColorClassifier
from ColorMaskGenerator import CMG
from SealLionLocations import Spots
import cv2
import json
import os
import pandas as pd
import pickle


def genSpots(reset=False, parellel=False):
    spotL = Spots(ResetJson=reset, parellel=parellel)
    spotL.run()


def genMaskedImage(ColorClassifier, imgName):
    mask = CMG(ColorClassifier)
    img = cv2.imread("./Train/" + imgName)
    img = mask.getMaskImg(img)
    cv2.imwrite("./MaskedImages/" + imgName, img)
    return 0


def gencolorclassifier(parellel=False, cropParam=3, sizeParam=100):
    if "SeaLionLoc.json" in os.listdir("."):
        fp = open("SeaLionLoc.json", 'r')
        SLL = json.load(fp)
        fp.close()
    else:
        return

    CC = ColorClassifier(SLL, parellel=parellel)
    CC.buildLionColors(cropParam=cropParam, sizeParam=sizeParam)
    print("Built LionColor Matrix")
    CC = CC.getLogitClassifier()
    print("Trained Color Classifier")
    fp = open("SeaLionCC.pickle", 'w')
    pickle.dump(CC, fp)
    fp.close()
    print("Dumped Pickle")


def genjson(parellel=False, cropParam=60, colorClassifier=None):
    if "SeaLionLoc.json" in os.listdir("."):
        fp = open("SeaLionLoc.json", 'r')
        SLL = json.load(fp)
        fp.close()
    else:
        print("No existing file for SeaLionLoc.json in working directory")
        return

    generator = GenJSON(SLL, cropParam=cropParam, colorClassifier=colorClassifier)
    generator.run(parellel=parellel)
    fp = open("SeaLionB.json", 'w')
    json.dump(generator.run(), fp)
    fp.close()
    print("Json Dump of the bounding rectangles")


def checkLocJson():
    if "SeaLionLoc.json" in os.listdir("."):
        fp = open("SeaLionLoc.json", 'r')
        results = json.load(fp)
        fp.close()
    else:
        print("No SeaLionLoc.json file in working Directory")
        return None
    if "train.csv" in os.listdir("."):
        trainCSV = pd.read_csv("train.csv")
    else:
        print("no train.csv file in working directory")
        return None

    total = 0
    for pic in results.keys():
        trainkey = int(pic.split(".")[0])
        for key in results[pic].keys():
            if key != 'error':
                total += abs(len(results[pic][key].keys()) - trainCSV[key][trainkey])

    print ("Total Number of MisClassifications: " + str(total))


def genBoxedImage(name):
    if "SeaLionB.json" in os.listdir("."):
        fp = open("SeaLionB.json", 'r')
        results = json.load(fp)
        fp.close()
    else:
        print("No SeaLionB.json File in current directory")

    imgDot = cv2.imread("./TrainDotted/" + name)
    for key in results[name].keys():
        for box in results[name][key].keys():
            nextBox = results[name][key][box]
            cv2.rectangle(imgDot, (nextBox[0][1], nextBox[0][0]), (nextBox[1][1], nextBox[1][0]), (255, 0, 0))

    cv2.imwrite("./TrainBoxed/" + name, imgDot)

