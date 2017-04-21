import json
import cv2
from BlobDetector import BlobDetector
import os
from multiprocessing import Process, Queue


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
        self.blobD = BlobDetector(5)

    def run(self):
        names = os.listdir("./Train/")
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
            for name in os.listdir("./Train/"):
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
                       'juveniles': {}, 'pups': {}, 'error': {}}
            for spot in spots:
                thisImg[self.getSpotColor(imgDot[spot[0]][spot[1]])][str((spot[0], spot[1]))] = (int(spot[0]), int(spot[1]))

            if mpq is None:
                self.jsonL[name] = thisImg
            else:
                mpq.put((name, thisImg))
        return None

    def getSpotColor(self, color):
        b, g, r = color
        if r > 150 and g < 50 and b < 50: # RED
            return 'adult_males'
        elif r > 150 and  b > 150: # MAGENTA
            return 'subadult_males'
        elif r < 100 and g > 150 and b < 100: # GREEN
            return 'pups'
        elif r < 80 and  g < 80 and b > 150: # BLUE
            return 'juveniles'
        elif r < 150 and g < 100 and b < 100:  # BROWN
            return 'adult_females'
        else:
            return 'error'

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