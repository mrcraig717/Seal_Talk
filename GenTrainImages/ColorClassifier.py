from sklearn.svm import SVC
from SealLogit import SealLogit
import numpy as np
import cv2
import os
import multiprocessing as ps


class ColorClassifier(object):

    def __init__(self, SLL, parellel=False):
        self.SLL = SLL
        self.parellel = parellel
        self.LionColors = np.zeros((1, 4))

    def parellelPSpawn(self, name, spots, cropParam, sizeParam, mpq):

        print("Processing image: " + name)
        img = cv2.imread("./Train/" + name)
        for key in spots.keys():
            if key != 'pups' and key != 'error':
                self.addLionColors(img, spots[key], cropParam, sizeParam)

        mpq.put(self.LionColors)
        print(name + " finished")

    def buildLionColors(self, names=None, cropParam=2, sizeParam=10):
        if names is None:
            names = os.listdir("./Train/")
        if ".gitignore" in names:
            names.remove(".gitignore")

        if self.parellel is True:
            procList = []
            mpq = ps.Queue()
            for i in range(4):
                name = names.pop()
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
                                name = names.pop()
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
                img = cv2.imread("./Train/" + name)
                print("Processing Image: " + name)
                for key in self.SLL[name].keys():
                    if key != 'pups' and key != 'error':
                        self.addLionColors(img, self.SLL[name][key], cropParam, sizeParam)

    def getRandomDraw(self, size, choose):
        perm = []
        choices = range(size)
        if choose > size:
            return choices
        for i in range(choose):
            perm.append(choices.pop(int(np.floor(np.random.rand() * len(choices)))))
        return perm

    def getClassifier(self):

        if self.LionColors is None:
            return None
        else:
            classifier = SVC()
            classifier.fit(self.LionColors[:, :-1], self.LionColors[:, -1:].ravel())
            return classifier

    def getLogitClassifier(self):
        if self.LionColors is None:
            return None
        else:
            classifier = SealLogit()
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
                    new = np.hstack((img[j][k], 1.0))
                    #if self.checkExistance(new) is False:
                    self.LionColors = np.vstack((self.LionColors, new))

        if len(lionSpots.keys()) < sizeParam:
            randPoints = np.random.rand(len(lionSpots.keys()) * 2)
        else:
            randPoints = np.random.rand(sizeParam * 2)

        for i in range(len(randPoints) // 2):
            spot = (int(np.floor(randPoints[i] * imgShape[0])), int(np.floor(randPoints[i + len(randPoints) // 2] * imgShape[1])))
            bounds = self.getIndexBounds(spot, imgShape, cropParam)
            for j in xrange(bounds[0][0], bounds[0][1] + 1, 1):
                for k in xrange(bounds[1][0], bounds[1][1] + 1, 1):
                    new = np.hstack((img[j][k], 0.0))
                    #if self.checkExistance(new) is False:
                    self.LionColors = np.vstack((self.LionColors, new))

    def checkExistance(self, new):
        for color in self.LionColors:
            if np.sum(color == new) == 4:
                return True

        return False
