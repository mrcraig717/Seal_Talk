import cv2
import os
import numpy as np
from multiprocessing import Process, Queue
from FloodFillLion import FillFloodLion


class GenJSON(object):

    def __init__(self, SLL, cropParam=100, colorClassifier=None):

        self.cropParam = cropParam
        self.SLL = SLL
        self.ffLion = FillFloodLion(colorClassifier=colorClassifier)
        self.jsonFile = {}
        self.CC = colorClassifier

    def parellelSpawn(self, name, mpq):
        img = cv2.imread("./Train/" + name)
        thisImg = {}
        for key in self.SLL[name].keys():
            thisClass = {}
            for spot in self.SLL[name][key]:
                crop = self.getCrop(img, self.SLL[name][key][spot])
                thisClass[spot] = self.retranslate(self.ffLion.genBoundingRectangle(crop))

            thisImg[key] = thisClass
        mpq.put((name, thisImg))

    def run(self, names=None, parellel=False):

        if names is None:
            names = os.listdir("./Train/")

        if parellel is True:
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
                        procList[0].join(5)
                else:
                    name, BoundBoxs = mpq.get()
                    self.jsonFile[name] = BoundBoxs

        else:
            pass
        return self.jsonFile

    def retranslate(self, subBoundRect):
        Xmax = self.currSpot[0] - self.shifts[0][0] + subBoundRect[0][0]
        Ymax = self.currSpot[1] - self.shifts[1][0] + subBoundRect[0][1]
        Xmin = self.currSpot[0] - self.shifts[0][0] + subBoundRect[1][0]
        Ymin = self.currSpot[1] - self.shifts[1][0] + subBoundRect[1][1]

        return [(Xmax, Ymax),(Xmin, Ymin)]

    def getCrop(self, img, spot):
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



