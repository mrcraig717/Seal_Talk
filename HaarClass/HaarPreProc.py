import os
import cv2
import numpy as np


class HaarPreProc:
	def __init__(self, foldImagesIn, BBdata, CMG, include=None, imagesToUse = 10):

		##############################################################################
		# String to path of the folder the Images are in
		# BBdata is the a json generated from GenJSON class with the bounding box information
		# CMG is the object used to house the color masking that was being applied for some of the training
		# imagestouse: if greater then number of files in the BB data will use all files in the BBdata json
		##############################################################################

		if include is None:
			self.include = ["adult_males", "subadult_males", "adult_females", "juveniles"]
		else:
			self.include = include
		if "posimg" in os.listdir("."):
			os.system("rm -r posimg")
		os.system("mkdir posimg")
		self.imagesToUse = imagesToUse

		if "negimg" in os.listdir("."):
			os.system("rm -r negimg")
		os.system("mkdir negimg")
		
		self.CMG = CMG
		self.names = []
		for name in BBdata.keys():
			if name in os.listdir(foldImagesIn):
				self.names.append(name)
		if len(self.names) < self.imagesToUse:
			self.imagesToUse = len(self.names)
		
		print("Using " + str(self.imagesToUse) + " total images for processing")
		self.BBdata = BBdata
		self.foldImagesIn = foldImagesIn
		self.posFiles = []
		self.background = None
		self.nNEGSamples = 0
		self.nPOSSamples = np.zeros(4)
		self.IMAGEPART = 512
	

	def run(self):
		for classtype in ["adult_males", "subadult_males", "adult_females", "juveniles"]:
			self.posFiles.append(open(classtype + ".txt", 'w'))

		self.background = open("bg.txt", 'w')
		self.popPosImgFolder()
		###############Use intensites as classes
		#self.meanInt, self.STD = self.getIntensitiePartitions()
		for name in self.names[:self.imagesToUse]:
			img = cv2.imread("./posimg/" + name)
			self.genNegImg(name, img)
			self.genPosImg(name, img)
		
		self.background.close()
		background = open("bg.txt", 'r')
		finalBack = background.read()[:-1]
		background.close()
		background = open("bg.txt", 'w')
		background.write(finalBack)
		background.close()
		####Work around for preventing parser errors in opencv train cascade
		for fil in self.posFiles:
			fil.close()
		for classtype in ["adult_males", "subadult_males", "adult_females", "juveniles"]:
			positive = open(classtype + ".txt", 'r')		
			finalPos = positive.read()[:-1]
			positive.close()
			positive = open(classtype + ".txt", 'w')
			positive.write(finalPos)
			positive.close()
		
		return self.nPOSSamples, self.nNEGSamples #, self.meanInt, self.STD

	def spotsInSub(self, name, subImg):
		######Determines if there exists a dot in the subimage being considered as a negative image
		for key in self.BBdata[name].keys():
			if key in self.include:
				for spot in self.BBdata[name][key].keys():
					nextS = self.BBdata[name][key][spot]
					####Check to see if the spot is inside our subImg
					if nextS[0] > subImg[0][0] and nextS[0] < subImg[0][1] and nextS[1] > subImg[1][0] and nextS[1] < subImg[1][1]:
						return [1]
		return None

	def genNegImg(self, name, img):
		###### Helper function used in generating the backgroud.txt file for the opencv Cascade Training 
		imgShape = np.shape(img)
		print("Partitioning Image: " + name)
		for i in xrange(1, imgShape[0] // self.IMAGEPART, 1):
			Xshift = i * self.IMAGEPART
			for j in xrange(1, imgShape[1] // self.IMAGEPART, 1):
				Yshift = j * self.IMAGEPART
				spots = self.spotsInSub(name, [(Xshift - self.IMAGEPART, Xshift),(Yshift - self.IMAGEPART, Yshift)])
				if spots is None:
					subImg =  img[Xshift - self.IMAGEPART:Xshift, Yshift - self.IMAGEPART:Yshift]   
					if np.mean(subImg) > 0:
						cv2.imwrite("./negimg/" + name.split(".")[0] + str(i) + str(j) + ".jpg", subImg)
						self.background.write("negimg/" + name.split(".")[0] + str(i) + str(j) +".jpg")
						self.nNEGSamples += 1
						self.background.write("\n")

	def popPosImgFolder(self):
		######3 allpies the function in 
		for name in self.names[:self.imagesToUse]:
			print("Processing Image: " + name)
			img = cv2.imread(self.foldImagesIn + name)
			img = self.CMG.getPlayImg(img)
			cv2.imwrite("./posimg/" + name, img)	


	def genPosImg(self, name, img):
		####### Generates the Text file that is used for opencv Casacade Training
		AM = 0
		SAM = 0
		AF = 0
		J = 0
		goodKeys = []
		for key in self.BBdata[name].keys():
			if key in self.include:
				for key2 in self.BBdata[name][key].keys():
					spot = self.BBdata[name][key][key2]
					if int(spot[0][0]) - int (spot[1][0]) > 30:
						crop = img[spot[1][0]:spot[0][0], spot[1][1]:spot[0][1]]
						#cropMean = np.mean(crop)
						if key == "adult_males" or key == "subadult_males":#cropMean is not np.nan:
							if True: #cropMean >= self.meanInt:
								if True: #cropMean >= self.meanInt + self.STD:
									goodKeys.append((0, key2, key))
									self.nPOSSamples[0] += 1
									AM += 1
								else:
									goodKeys.append((1, key2, key))
									self.nPOSSamples[1] += 1
									SAM += 1
							else:
								if cropMean <= self.meanInt - self.STD:
									goodKeys.append((3, key2, key))
									self.nPOSSamples[3] += 1
									J += 1
								else:
									goodKeys.append((2, key2, key))
									self.nPOSSamples[2] += 1
									AF += 1
		if AM != 0:
			self.posFiles[0].write("posimg/" + name + " " + str(AM) + " ")
		if SAM != 0:
			self.posFiles[1].write("posimg/" + name + " " + str(SAM) + " ")
		if AF != 0:
			self.posFiles[2].write("posimg/" + name + " " + str(AF) + " ")
		if J != 0:
			self.posFiles[3].write("posimg/" + name + " " + str(J) + " ")
		
		for goodkey in goodKeys:
			spot = self.BBdata[name][goodkey[-1]][goodkey[1]]
			self.posFiles[goodkey[0]].write(str(spot[1][1]) + " " + str(spot[1][0]) + " " + str(spot[0][1] - spot[1][1]) + " " + str(spot[0][0] - spot[1][0]) + "  ")
		
		if AM != 0:
			self.posFiles[0].write("\n")
		if SAM != 0:
			self.posFiles[1].write("\n")
		if AF != 0:
			self.posFiles[2].write("\n")
		if J != 0:
			self.posFiles[3].write("\n")
	
	def getIntensitiePartitions(self):
		############## Was being used at one point to sort the avg intensities of the positive images
		intensities = []
		for name in self.names[:self.imagesToUse]:
			print("Processing Image Intensities: " + name)
			img = cv2.imread("./posimg/" + name)
			print(np.shape(img))
			for key in self.BBdata[name].keys():
				if key in self.include:
					for key2 in self.BBdata[name][key].keys():
						spot = self.BBdata[name][key][key2]
						if int(spot[0][0]) - int (spot[1][0]) > 30:
							crop = img[spot[1][0]:spot[0][0], spot[1][1]:spot[0][1]]
							MEAN = np.mean(crop)
							if MEAN is np.nan:
								pass
							else:
								intensities.append(MEAN)
		return np.mean(intensities), np.std(intensities)