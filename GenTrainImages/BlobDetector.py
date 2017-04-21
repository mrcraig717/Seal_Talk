import numpy as np


class BlobDetector(object):

    def __init__(self, threshold):
        self.threshold = threshold
        self.blobs = None
        self.imgShape = None

    def detect(self, img):
        self.blobs = []
        self.imgShape = np.shape(img)
        for i in range(self.imgShape[0]):
            for j in range(self.imgShape[1]):
                if img[i][j] == 255:
                    img = self.fillBlob(img, (i, j))

        return self.blobs

    def fillBlob(self, img, seed):
        Queue = [seed]
        inBlob = []
        while Queue:
            on = Queue.pop()
            if img[on[0]][on[1]] == 255:
                inBlob.append(on)
                img[on[0]][on[1]] = 100
                if on[0] + 1 < self.imgShape[0] and img[on[0] + 1][on[1]] != 100:
                    Queue.append((on[0] + 1, on[1]))
                if on[0] - 1 >= 0 and img[on[0] - 1][on[1]] != 100:
                    Queue.append((on[0] - 1, on[1]))
                if on[1] + 1 < self.imgShape[1] and img[on[0]][on[1] + 1] != 100:
                    Queue.append((on[0], on[1] + 1))
                if on[1] - 1 >= 0 and img[on[0]][on[1] - 1] != 100:
                    Queue.append((on[0], on[1] - 1))

        if len(inBlob) > self.threshold:
            self.blobs.append(self.getCentroid(inBlob))
        return img

    def getCentroid(self, inBlob):
        result = np.zeros((2))
        for point in inBlob:
            result += np.array(point, dtype=float)
        result /= float(len(inBlob))
        return np.array((int(result[0]), int(result[1])))
