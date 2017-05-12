import numpy as np


class FillFloodLion(object):

    def __init__(self, colorClassifier=None, include = ["adult_males", "subadult_males", "adult_females", "juveniles", "pups"]):
        self.CC = colorClassifier
        self.newColor = np.array([255, 0, 255], dtype='uint8')
        self.plotFlag = True
        self.interestColor = np.array([0,0,255], dtype='uint8')
        self.include = include
    
    def genBoundingRectangle(self, img, classtype):

        self.imgShape = np.shape(img[0])
        seed = img[1]
        img = img[0]
        Xmax = seed[0]
        Xmin = seed[0]
        Ymax = seed[1]
        Ymin = seed[1]
        
        if classtype in self.include:

            self.seedColor = img[seed[0]][seed[1]].astype(int)
            nextQueue = [seed]
            maxQueueSize = 0
            totalFound = 0
            while nextQueue:
                if len(nextQueue) > maxQueueSize:
                    maxQueueSize = len(nextQueue)
                else:
                    if maxQueueSize > 20 and len(nextQueue) < maxQueueSize // 4:
                        break

                Queue = nextQueue
                nextQueue = []
                while Queue:
                    on = Queue.pop()
                        
                    if on[0] + 1 < self.imgShape[0] and self.isValidColor(img[on[0] + 1][on[1]]) or totalFound < 50 and np.sum(img[on[0] + 1][on[1]] == self.newColor) != 3:
                        img[on[0] + 1][on[1]] = self.newColor
                        nextQueue.append((on[0] + 1, on[1]))
                        if on[0] + 1 > Xmax:
                            Xmax = on[0] + 1
                        totalFound += 1
                    if on[0] - 1 >= 0 and self.isValidColor(img[on[0] - 1][on[1]]) or totalFound < 50 and np.sum(img[on[0] - 1][on[1]] == self.newColor) != 3:
                        img[on[0] - 1][on[1]] = self.newColor
                        nextQueue.append((on[0] - 1, on[1]))
                        if on[0] - 1 < Xmin:
                            Xmin = on[0] - 1
                        totalFound += 1
                    if on[1] + 1 < self.imgShape[1] and self.isValidColor(img[on[0]][on[1] + 1]) or totalFound < 50 and np.sum(img[on[0]][on[1] + 1] == self.newColor) != 3:
                        img[on[0]][on[1] + 1] = self.newColor
                        nextQueue.append((on[0], on[1] + 1))
                        if on[1] + 1 > Ymax:
                            Ymax = on[1] + 1
                        totalFound += 1
                    if on[1] - 1 >= 0 and self.isValidColor(img[on[0]][on[1] - 1]) or totalFound < 50 and np.sum(img[on[0]][on[1] - 1] == self.newColor) != 3:
                        img[on[0]][on[1] - 1] = self.newColor
                        nextQueue.append((on[0], on[1] - 1))
                        if on[1] - 1 < Ymin:
                            Ymin = on[1] - 1
                        totalFound += 1
                        
                if totalFound < 50:
                    maxQueueSize = len(nextQueue)
        
        if (Xmax - Xmin > 40 or Ymax - Ymin > 40) and (classtype == 'subadult_males' or classtype == 'adult_males'):

            extendBy = 10
            if Ymax + extendBy < self.imgShape[1]:
                Ymax += extendBy
            else:
                Ymax = self.imgShape[1] - 1
            if Ymin - extendBy > 0:
                Ymin -= extendBy
            else:
                Ymin = 0

            if Xmax + extendBy < self.imgShape[0]:
                Xmax += extendBy
            else:
                Xmax = self.imgShape[0] - 1
            if Xmin - extendBy > 0:
                Xmin -= extendBy
            else:
                Xmin = 0
        elif (Xmax - Xmin > 20 or Ymax - Ymin > 20) and (classtype == 'adult_females' or classtype == 'juveniles'): 
            extendBy = 5
            if Ymax + extendBy < self.imgShape[1]:
                Ymax += extendBy
            else:
                Ymax = self.imgShape[1] - 1
            if Ymin - extendBy > 0:
                Ymin -= extendBy
            else:
                Ymin = 0

            if Xmax + extendBy < self.imgShape[0]:
                Xmax += extendBy
            else:
                Xmax = self.imgShape[0] - 1
            if Xmin - extendBy > 0:
                Xmin -= extendBy
            else:
                Xmin = 0
        else:
            extendBy = 5
            if seed[1] + extendBy < self.imgShape[1]:
                Ymax = seed[1] + extendBy
            else:
                Ymax = self.imgShape[1] - 1
            
            if seed[1] - extendBy > 0:
                Ymin = seed[1] - extendBy
            else:
                Ymin = 0

            if seed[0] + extendBy < self.imgShape[0]:
                Xmax = seed[0] + extendBy
            else:
                Xmax = self.imgShape[0] - 1
            if seed[0] - extendBy > 0:
                Xmin = seed[0] - extendBy
            else:
                Xmin = 0

        return [(Xmax, Ymax), (Xmin, Ymin)]


    def isValidColor(self, color):

        if np.sum(color == self.newColor) == 3:
            return False
        
        elif self.CC is None:
            if np.sum(abs(color.astype(int) - self.seedColor)) < 50:
                return True
            else:
                return False
        else:
            if self.CC.predict((color.astype(float)).reshape(1, -1))[0] == 1.0:
                return True
            else:
                return False