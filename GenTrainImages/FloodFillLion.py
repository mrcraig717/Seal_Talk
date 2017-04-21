import numpy as np


class FillFloodLion(object):

    def __init__(self, colorClassifier=None):
        self.CC = colorClassifier
        self.newColor = np.array([255, 0, 255], dtype='uint8')
        self.plotFlag = True

    def genBoundingRectangle(self, img):

        self.imgShape = np.shape(img[0])
        seed = img[1]
        img = img[0]
        Xmax = seed[0]
        Xmin = seed[0]
        Ymax = seed[1]
        Ymin = seed[1]
        self.seedColor = img[seed[0]][seed[1]].astype(int)
        nextQueue = [seed]
        onQueueSize = 0
        while nextQueue:
            if len(nextQueue) > onQueueSize:
                onQueueSize = len(nextQueue)
            else:
                break

            Queue = nextQueue
            nextQueue = []
            while Queue:
                on = Queue.pop()
                if on[0] + 1 < self.imgShape[0] and self.isValidColor(img[on[0] + 1][on[1]]):
                    img[on[0] + 1][on[1]] = self.newColor
                    nextQueue.append((on[0] + 1, on[1]))
                    if on[0] + 1 > Xmax:
                        Xmax = on[0] + 1

                if on[0] - 1 >= 0 and self.isValidColor(img[on[0] - 1][on[1]]):
                    img[on[0] - 1][on[1]] = self.newColor
                    nextQueue.append((on[0] - 1, on[1]))
                    if on[0] - 1 < Xmin:
                        Xmin = on[0] - 1

                if on[1] + 1 < self.imgShape[1] and self.isValidColor(img[on[0]][on[1] + 1]):
                    img[on[0]][on[1] + 1] = self.newColor
                    nextQueue.append((on[0], on[1] + 1))
                    if on[1] + 1 > Ymax:
                        Ymax = on[1] + 1

                if on[1] - 1 >= 0 and self.isValidColor(img[on[0]][on[1] - 1]):
                    img[on[0]][on[1] - 1] = self.newColor
                    nextQueue.append((on[0], on[1] - 1))
                    if on[1] - 1 < Ymin:
                        Ymin = on[1] - 1

        extendBy = 20
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