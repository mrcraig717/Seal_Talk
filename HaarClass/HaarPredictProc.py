import numpy as np
import cv2 
import sys
sys.path.append("../")
from GenTrainImages import ColorMaskGenerator
##########################################################################
# Class used to handle the Haar Predict Process
# putting the whole does to work do to its size
##########################################################################

class HaarPredictProc:
	def __init__(self, img, classifiers, cropParam):
		self.img = img
		self.classifiers = classifiers
		self.cropParam = cropParam
		self.imgShape = np.shape(img)
		self.currSpot = None
		self.shifts = None
		self.CMG = ColorMaskGenerator.CMG(None)
		self.img = img 

	def run(self, minBoxSize, maxBoxSize, numberNeibhors):
		onX = self.cropParam
		
		while onX < self.imgShape[0]:
			onY = self.cropParam
			print("hit bottom of pic")
			while onY < self.imgShape[1]:
				print("Predicting on Sub Pic")
				crop = self.getCrop((onX, onY))
				crop = self.CMG.getPlayImg(crop)
				for i in range(len(self.classifiers)):
					subBoundRects = self.classifiers[i].detectMultiScale(crop, scaleFactor=1.1, minNeighbors=numberNeibhors, 
																	minSize=minBoxSize, maxSize=maxBoxSize)
					print(str(len(subBoundRects)) + " sealions found")
					for rect in subBoundRects:
						subRect = [[rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1]]]
						nextBox = self.retranslate(subRect)
						cv2.rectangle(self.img, (nextBox[0][1], nextBox[0][0]), (nextBox[1][1], nextBox[1][0]), (0, 0, 255))
				onY += self.cropParam
			onX += self.cropParam
		return self.img

	def retranslate(self, subBoundRect):
		Xmax = self.currSpot[0] - self.shifts[0][0] + subBoundRect[0][0]
		Ymax = self.currSpot[1] - self.shifts[1][0] + subBoundRect[0][1]
		Xmin = self.currSpot[0] - self.shifts[0][0] + subBoundRect[1][0]
		Ymin = self.currSpot[1] - self.shifts[1][0] + subBoundRect[1][1]
		return [(Xmax, Ymax),(Xmin, Ymin)]

	def getCrop(self, spot):
		self.imgShape = np.shape(self.img)
		self.currSpot = spot
		self.shifts = [[None, None],[None, None]]
		if spot[0] + self.cropParam < self.imgShape[0]:
			self.shifts[0][1] = self.cropParam
		else:
			self.shifts[0][1] = self.imgShape[0] - spot[0] - 1
		if spot[0] - self.cropParam >= 0:
			self.shifts[0][0] = self.cropParam
		else:
			self.shifts[0][0] = spot[0]
		
		if spot[1] + self.cropParam < self.imgShape[1]:
			self.shifts[1][1] = self.cropParam
		else:
			self.shifts[1][1] = self.imgShape[1] - spot[1] - 1
		
		if spot[1] - self.cropParam >= 0:
			self.shifts[1][0] = self.cropParam
		else:
			self.shifts[1][0] = spot[1]

		return self.img[spot[0] - self.shifts[0][0]:spot[0] + self.shifts[0][1], 
		                spot[1] - self.shifts[1][0]:spot[1] + self.shifts[1][1]]

