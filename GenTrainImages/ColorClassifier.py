from sklearn.svm import SVC
from SealLogit import seallogit
import numpy as np
import cv2
import os
import multiprocessing as ps
#############################
# Class used for collection negative and positive samples fro classifiying Sea Lion Colors 
#########################
class ColorClassifier(object):

    def __init__(self, SLL, parellel=False, include = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]):
        self.SLL = SLL
        self.parellel = parellel
        self.LionColors = np.zeros((1, 4))
        self.LionColors = np.vstack((self.LionColors, np.ones((1,4))))
        self.include = include

    def parellelPSpawn(self, name, spots, cropParam, sizeParam, mpq):

        print("Processing image: " + name)
        img = cv2.imread("./MaskedImages/" + name)
        totalPos = 0
        for key in spots.keys():
            if key in self.include:
                self.addLionColors(img, spots[key], cropParam, sizeParam)
                if sizeParam > len(spots[key].keys()):
                    totalPos += len(spots[key].keys())
                else:
                    totalPos += sizeParam

        
        self.addNegSamples(img, cropParam, totalPos)
        mpq.put(self.LionColors)
        print(name + " finished")

    def buildLionColors(self, names=None, cropParam=2, sizeParam=100):
        if names is None:
            names = self.SLL.keys()
       
        if self.parellel is True:
            procList = []
            mpq = ps.Queue()
            for i in range(4):
                name = names.pop().split("/")[-1]
                procList.append(ps.Process(target=self.parellelPSpawn,
                                           args=(name, self.SLL[name], cropParam, sizeParam, mpq,)))
                procList[-1].start()

            while procList:
                if mpq.empty() is True:
                    for prc in procList:
                        if prc.is_alive():
                            pass
                        else:
                            procList.remove(prc)
                            if names:
                                name = names.pop().split("/")[-1]
                                procList.append(ps.Process(target=self.parellelPSpawn,
                                                           args=(name, self.SLL[name], cropParam, sizeParam, mpq,)))
                                procList[-1].start()

                    if procList:
                        procList[0].join(5)
                else:
                    lColor = mpq.get()
                    self.LionColors = np.vstack((self.LionColors, lColor))

        else:
            for name in names:
                img = cv2.imread("./MaskedImages/" + name)
                print("Processing Image: " + name)
                for key in self.SLL[name].keys():
                    if key in self.include:
                        self.addLionColors(img, self.SLL[name][key], cropParam, sizeParam)
                        
                self.addNegSamples(img, cropParam, sizeParam)
    def getRandomDraw(self, size, choose):
        perm = []
        choices = range(size)
        if choose > size:
            return choices
        for i in range(choose):
            perm.append(choices.pop(int(np.floor(np.random.rand() * len(choices)))))
        return perm

    def getSVMClassifier(self):

        if np.shape(self.LionColors)[0] < 2:
            return None
        else:
            classifier = SVC()
            classifier.fit(self.LionColors[:, :-1], self.LionColors[:, -1:].ravel())
            return classifier

    def getLogitClassifier(self):
        #####Returns a Logistic Classifier trained with the colors collected after running build Lion Colors 
        if np.shape(self.LionColors)[0] < 2:
            return None
        else:
            classifier = seallogit()
            classifier.fit(self.LionColors[:, :-1], self.LionColors[:, -1:].ravel())
            return classifier


    def getIndexBounds(self, spot, imgShape, cropParam):

        bounds = [[None, None], [None, None]]
        if spot[0] + cropParam < imgShape[0]:
            bounds[0][1] = spot[0] + cropParam
        else:
            bounds[0][1] = imgShape[0] - cropParam
        if spot[0] - cropParam > 0:
            bounds[0][0] = spot[0] - cropParam
        else:
            bounds[0][0] = 0
        if spot[1] + cropParam < imgShape[1]:
            bounds[1][1] = spot[1] + cropParam
        else:
            bounds[1][1] = imgShape[0] - cropParam

        if spot[1] - 5 > 0:
            bounds[1][0] = spot[1] - 5
        else:
            bounds[1][0] = 0

        return bounds

    def releaseFitMatrix(self):
        self.LionColors = None

    def addLionColors(self, img, lionSpots, cropParam, sizeParam):
        imgShape = np.shape(img)

        for i in self.getRandomDraw(len(lionSpots.keys()), sizeParam):
            spot = lionSpots[lionSpots.keys()[i]]
            bounds = self.getIndexBounds(spot, imgShape, cropParam)
            for j in xrange(bounds[0][0], bounds[0][1] + 1, 1):
                for k in xrange(bounds[1][0], bounds[1][1] + 1, 1):
                    if np.sum(img[j][k]) > 200:
                        new = np.hstack((img[j][k], 1.0))
                        self.LionColors = np.vstack((self.LionColors, new))


    def addNegSamples(self, img, cropParam, sizeParam):
        imgShape = np.shape(img)
        randPoints = np.random.rand(sizeParam * 2)
        for i in range(len(randPoints) // 2):
            spot = (int(np.floor(randPoints[i] * imgShape[0])), int(np.floor(randPoints[i + len(randPoints) // 2] * imgShape[1])))
            bounds = self.getIndexBounds(spot, imgShape, cropParam)
            for j in xrange(bounds[0][0], bounds[0][1] + 1, 1):
                for k in xrange(bounds[1][0], bounds[1][1] + 1, 1):
                    new = np.hstack((img[j][k], 0.0))
                    self.LionColors = np.vstack((self.LionColors, new))        