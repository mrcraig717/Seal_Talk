from GenJSON import GenJSON
from ColorClassifier import ColorClassifier
from ColorMaskGenerator import CMG
from PartitionImg import GenGoogleNetPart
from SealLionLocations import Spots
import numpy as np
import cv2
import json
import os
import pandas as pd
import pickle



def genTrainPFolder():
    ####Utilitite used for creating the partioned images for detect net
    gen = GenGoogleNetPart()
    gen.run()

def genSpots(reset=False, parellel=False):
    ######Utilite used for generating the SeaLionLocations.
    spotL = Spots(ResetJson=reset, parellel=parellel)
    spotL.run()

def gencolorclassifier(parellel=False, cropParam=3, sizeParam=100, MaggieLoop=False):
    if "SeaLionLoc.json" in os.listdir("."):
        fp = open("SeaLionLoc.json", 'r')
        SLL = json.load(fp)
        fp.close()
    else:
        print("No SealLionLoc.json existing in current directory")
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


def genjson(parellel=False, cropParam=60, colorClassifier=None, include = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]):
    
    #####Utilities Function used for high level call to generate the SeaLionB.json file or the Bounding boxes if you will
    if "SeaLionLoc.json" in os.listdir("."):
        fp = open("SeaLionLoc.json", 'r')
        SLL = json.load(fp)
        fp.close()
    else:
        print("No existing file for SeaLionLoc.json in working directory")
        return

    generator = GenJSON(SLL, cropParam=cropParam, colorClassifier=colorClassifier, include = include)
    fp = open("SeaLionB.json", 'w')
    json.dump(generator.run(parellel=parellel), fp)
    fp.close()
    print("Json Dump of the bounding rectangles")


def checkLocJson(LocFile= None):
    ########Funciton used to compare the results of the dot 
    ######## finding process to the train.csv file provided by Kaggle
    if LocFile is None:
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
    
    finalJson = {}
    total = 0
    totalSquare = 0

    for pic in results.keys():
        trainkey = int(pic.split(".")[0])
        dropPic = False
        
        for key in results[pic].keys():
            print pic, key, trainCSV[key][trainkey], len(results[pic][key].keys())          
            difference = abs(len(results[pic][key].keys()) - trainCSV[key][trainkey]) 
            total += difference
            totalSquare += difference ** 2
            if abs(len(results[pic][key].keys()) - trainCSV[key][trainkey]) > 50:
                dropPic = True
        if dropPic is False:
            finalJson[pic] = results[pic]

    print ("Total Number of MisClassifications: " + str(total))
    print ("Total Number of Images Dotted Matched Original: " + str(len(finalJson.keys())))
    print ("Root mean square error: "  + str(np.sqrt(totalSquare) / float(len(finalJson.keys()))))

    fp = open("SeaLionLoc2.json", "w")
    json.dump(finalJson, fp)
    fp.close()


def genALLBoxedImages():
    if "SeaLionB.json" in os.listdir("."):
        fp = open("SeaLionB.json", 'r')
        results = json.load(fp)
        fp.close()
    else:
        print("No SeaLionB.json File in current directory")

    for name in results.keys():
        imgDot = cv2.imread("./TrainDotted/" + name)
        for key in results[name].keys():
            for box in results[name][key].keys():
                nextBox = results[name][key][box]
                if key == "adult_males":
                    cv2.rectangle(imgDot, (nextBox[0][1], nextBox[0][0]), (nextBox[1][1], nextBox[1][0]), (0, 0, 255)) 
                elif key == "subadult_males":
                    cv2.rectangle(imgDot, (nextBox[0][1], nextBox[0][0]), (nextBox[1][1], nextBox[1][0]), (255, 0, 255)) 
                elif key == "adult_females":
                    cv2.rectangle(imgDot, (nextBox[0][1], nextBox[0][0]), (nextBox[1][1], nextBox[1][0]), (45, 45, 130)) 
                elif key == "juveniles":
                    cv2.rectangle(imgDot, (nextBox[0][1], nextBox[0][0]), (nextBox[1][1], nextBox[1][0]), (255, 0, 0))
                elif key == "pups":
                    cv2.rectangle(imgDot, (nextBox[0][1], nextBox[0][0]), (nextBox[1][1], nextBox[1][0]), (0, 255, 0))

        cv2.imwrite("./TrainBoxed/" + name, imgDot)

def genBoxedImage(name):
    if "SeaLionB.json" in os.listdir("."):
        fp = open("SeaLionB.json", 'r')
        results = json.load(fp)
        fp.close()
    else:
        print("No SeaLionB.json File in current directory")

    imgDot = cv2.imread("./MaskPlay/" + name)
    for key in results[name].keys():
        for box in results[name][key].keys():
            nextBox = results[name][key][box]
            cv2.rectangle(imgDot, (nextBox[0][1], nextBox[0][0]), (nextBox[1][1], nextBox[1][0]), (255, 0, 0))

    cv2.imwrite("./TrainBoxed/" + name, imgDot)


def genBoxedPImage(name):
    #####Utilitie once used for boxing the partitioned images for Detect Net these images are not in the git there to many fo them
    if "SeaLionB.json" in os.listdir("./TrainP"):
        fp = open("./TrainP/SeaLionB.json")
        results = json.load(fp)
        fp.close()
    else:
        print("No SeaLionB.json in TrainP Folder")

    imgDot = cv2.imread("./TrainP/" + name)
    for key in results[name].keys():
        for box in results[name][key].keys():
            nextBox = results[name][key][box]
            cv2.rectangle(imgDot, (nextBox[0][1], nextBox[0][0]), (nextBox[1][1], nextBox[1][0]), (255, 0, 0))

    cv2.imwrite("./TrainBoxed/" + name, imgDot)    