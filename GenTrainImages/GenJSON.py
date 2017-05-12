import cv2
import os
import numpy as np
from multiprocessing import Process, Queue
from FloodFillLion import FillFloodLion
from ColorClassifier import ColorClassifier
##########################################################################
#Class made for Generating Sea Lion Bounding Boxes
#args:
#   SLL: json file loaded with the locations of the Dots and there Classification
#   Bool: True -> only generate bounding boxes for images that don't exist in current Json
#         Fasle -> Will delete current JSON and start over
#   cropParam: parameter setting a max any given bounding box cropparam =100 results in a max = (400,400), Defualt 60
#   colorClassifier: None implies a color classifier will be create for each image. If one is pass it will be used for all images 
#   include: Which classes will be included in the process
##########################################################################


class GenJSON(object):

    def __init__(self, SLL, useExistingJSON=True, cropParam=60, colorClassifier=None,
                 include = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]):

        #####Color Classifier is dead Parameter now don't have to go back and change everything
        self.cropParam = cropParam
        self.SLL = SLL
        self.ffLion = FillFloodLion(colorClassifier=colorClassifier)
        if useExistingJSON is False:
            self.jsonFile = {}
        else:
            self.jsonFile = {}
        self.CC = colorClassifier
        self.include = include

    def parellelSpawn(self, name, mpq):
        ##########Function used as the target for paraelizing the Process for large
        ##########Data sets
        img = cv2.imread("./MaskedImages/" + name)
        thisImg = {}
        CC = ColorClassifier(self.SLL, include = self.include)
        CC.buildLionColors(names=[name], cropParam=3, sizeParam=100)
        thisFFLion = FillFloodLion(colorClassifier = CC.getLogitClassifier(), include=self.include)
        for key in self.SLL[name].keys():
            thisClass = {}
            for spot in self.SLL[name][key].keys():
                crop = self.getCrop(img, self.SLL[name][key][spot])
                thisClass[spot] = self.retranslate(thisFFLion.genBoundingRectangle(crop, key))

            thisImg[key] = thisClass
           
        mpq.put((name, thisImg))

    def run(self, names=None, parellel=False):
        #################################
        #Function for launch the process don't use parellel true if you are running on windows system
        # names = None then the process will run for all files in the Train folder
        # returns json file with bounding box information for all images in name List
        ###################################
        if names is None:
            names = self.SLL.keys()

        if parellel is True and len(names) > 3:
            mpq = Queue()
            procList = []
            for i in range(4):
                print("Generating Boxes for Image " + names[-1])
                procList.append(Process(target=self.parellelSpawn, args=(names.pop(), mpq,)))
                procList[-1].start()

            while procList:
                if mpq.empty() is True:
                    for prc in procList:
                        if prc.is_alive():
                            pass
                        else:
                            procList.remove(prc)
                            if names:
                                print("Generating Boxes for Image " + names[-1])
                                procList.append(Process(target=self.parellelSpawn,
                                                        args=(names.pop(), mpq,)))
                                procList[-1].start()

                    if procList:
                        procList[0].join(3)
                else:
                    name, BoundBoxs = mpq.get()
                    self.jsonFile[name] = BoundBoxs

        else:
            for name in names:
                print("Generating Boxes for Image: " + name)
                img = cv2.imread("./Train/" + name)
                thisImg = {}
                CC = ColorClassifier(self.SLL, include = self.include)
                CC.buildLionColors(names=[name], cropParam=3, sizeParam=100)
                thisFFLion = FillFloodLion(colorClassifier = CC.getLogitClassifier(), include=self.include)
                for key in self.SLL[name].keys():
                    thisClass = {}
                    for spot in self.SLL[name][key]:
                        crop = self.getCrop(img, self.SLL[name][key][spot])
                        thisClass[spot] = self.retranslate(thisFFLion.genBoundingRectangle(crop, key))

                    thisImg[key] = thisClass
                self.jsonFile[name] = thisImg


        return self.jsonFile

    def retranslate(self, subBoundRect):
        ###########Used to Translate the BB found in the Crop back to original Image
        Xmax = self.currSpot[0] - self.shifts[0][0] + subBoundRect[0][0]
        Ymax = self.currSpot[1] - self.shifts[1][0] + subBoundRect[0][1]
        Xmin = self.currSpot[0] - self.shifts[0][0] + subBoundRect[1][0]
        Ymin = self.currSpot[1] - self.shifts[1][0] + subBoundRect[1][1]

        return [(Xmax, Ymax),(Xmin, Ymin)]

    def getCrop(self, img, spot):
        ###########Pulls crop from Image around a spot based on crop parameter that was 
        imgShape = np.shape(img)
        self.currSpot = spot
        self.shifts = [[None, None],[None, None]]

        if spot[0] + self.cropParam < imgShape[0]:
            self.shifts[0][1] = self.cropParam
        else:
            self.shifts[0][1] = imgShape[0] - spot[0] - 1

        if spot[0] - self.cropParam >= 0:
            self.shifts[0][0] = self.cropParam
        else:
            self.shifts[0][0] = spot[0]

        if spot[1] + self.cropParam < imgShape[1]:
            self.shifts[1][1] = self.cropParam
        else:
            self.shifts[1][1] = imgShape[1] - spot[1]

        if spot[1] - self.cropParam >= 0:
            self.shifts[1][0] = self.cropParam
        else:
            self.shifts[1][0] = spot[1]

        return img[spot[0] - self.shifts[0][0]:spot[0] + self.shifts[0][1],
                   spot[1] - self.shifts[1][0]:spot[1] + self.shifts[1][1]], (self.shifts[0][0], self.shifts[1][0])



