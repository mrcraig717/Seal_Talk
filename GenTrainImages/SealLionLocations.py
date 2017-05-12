import json
import cv2
from BlobDetector import BlobDetector
import os
from multiprocessing import Process, Queue
from sklearn import cluster
import pandas as pd
import numpy as np

miscatches = [3,7,9,21,30,34,71,81,89,97,151,184,215,234,242,268,290,311,331,344,380,384,406,421,469,475,490,499,507,530,531,605,607,614,621,638,644,687,712,721,767,779,781,794,800,811,839,840,869,882,901,903,905,909,913,927,946]

labels = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups" ]

class Spots:

    def __init__(self, ResetJson=False, parellel=False):

        self.parellel = parellel
        if ResetJson is True:
            self.jsonL = {}
            self.reset = True
        else:
            self.reset = False
            if "SeaLionLoc.json" in os.listdir("."):
                fp = open("SeaLionLoc.json", 'r')
                self.jsonL = json.load(fp)
                fp.close()
            else:
                self.jsonL = {}
        self.blobD = BlobDetector(20)
        self.trainCSV = pd.read_csv("./train.csv")
        
    def run(self):
        global miscatches
        names = os.listdir("./Train/")
        if ".gitignore" in names:
            names.remove(".gitignore")
        if ".gitignore~" in names:
            names.remove(".gitignore~")
        
        keepers = []
        for name in names:
            if int(name.split(".")[0]) in miscatches:
                pass
            else:
                keepers.append(name)
        names = keepers
        
        if self.reset is False:
            nextnames = []
            for name in names:
                if name in self.jsonL.keys():
                    pass
                else:
                    nextnames.append(name)
            names = nextnames

        if self.parellel is True:

            mpq = Queue()
            procList = []
            for i in range(4):
                print("Processing Image " + names[-1])
                procList.append(Process(target=self.getSpots, args=(names.pop(), mpq,)))
                procList[-1].start()
            while procList:
                if mpq.empty():
                    for proc in procList:
                        if proc.is_alive():
                            pass
                        else:
                            procList.remove(proc)
                            if names:
                                print("Processing Image  " + names[-1])
                                procList.append(Process(target=self.getSpots, args=(names.pop(), mpq,)))
                                procList[-1].start()
                    if procList:
                        procList[-1].join(5)

                else:
                    res = mpq.get()
                    self.jsonL[res[0]] = res[1]
        else:
            for name in names:
                print("Processing Image " + name)
                self.getSpots(name)
        fp = open("SeaLionLoc.json", 'w')
        json.dump(self.jsonL, fp)
        fp.close()

    def getSpots(self, name, mpq=None):

        if name in self.jsonL.keys():
            return None
        else:
            imgf, imgDot = self.getIMGF(name)
            spots = self.blobD.detect(imgf)
            thisImg = {'adult_males': {}, 'subadult_males': {}, 'adult_females': {},
                       'juveniles': {}, 'pups': {}}
            
            index = name.split(".")[0]
            SpotClassifier = self.getSpotColor(spots, imgDot, int(index))
            for i in range(len(spots)):
                thisImg[SpotClassifier[i]][str((spots[i][0], spots[i][1]))] = (int(spots[i][0]), int(spots[i][1]))

            if mpq is None:
                self.jsonL[name] = thisImg
            else:
                mpq.put((name, thisImg))
        return None

    def getSpotColor(self, spots, imgDot, index):
        global labels
        means = []
        for key in self.trainCSV.keys()[1:]:
            if self.trainCSV[key][index] != 0:
                means.append(1)
            else:
                means.append(0)

        startingCentroids = []
        finalclasses = []
        if means[0] == 1:
            startingCentroids.append([0,0,255])
            finalclasses.append(0)
        if means[1] == 1:
            startingCentroids.append([255, 0, 255])
            finalclasses.append(1)
        if means[2] == 1:
            startingCentroids.append([42, 42, 165])
            finalclasses.append(2)
        if means[3] == 1:
            startingCentroids.append([255, 0, 0])
            finalclasses.append(3)
        if means[4] == 1:
            startingCentroids.append([0, 255, 0])
            finalclasses.append(4)

        CC = cluster.KMeans(sum(means), n_init = 1, init=np.array(startingCentroids))
        X = []
        for spot in spots:
            X.append(imgDot[spot[0]][spot[1]].astype(float))
        X = np.array(X)
        CC.fit(X)
        result = []
        classes = CC.predict(X)
        
        for num in classes:
            result.append(labels[finalclasses[num]])
        return result


    def getIMGF(self, name):
        img = cv2.imread("./Train/" + name)
        imgDot = cv2.imread("./TrainDotted/" + name)
        imgf = cv2.absdiff(img, imgDot)
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask[mask < 20] = 0
        mask[mask > 0] = 255
        imgf = cv2.bitwise_or(imgf, imgf, mask=mask)
        mask = cv2.cvtColor(imgDot, cv2.COLOR_BGR2GRAY)
        mask[mask < 20] = 0
        mask[mask > 0] = 255
        imgf = cv2.bitwise_or(imgf, imgf, mask=mask)
        imgf = cv2.cvtColor(imgf, cv2.COLOR_BGR2GRAY)
        imgf[imgf < 20] = 0
        imgf[imgf > 0] = 255
        return imgf, imgDot
