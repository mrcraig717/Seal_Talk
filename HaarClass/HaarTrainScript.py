import os
import json
import cv2
import numpy as np

IMAGEPART = 256

background = open("bg.txt", 'w')
positive = open("sealions.txt" , 'w')


if 'negimg' in os.listdir("."):
	os.system("rm -r negimg")

os.system("mkdir negimg")

if 'posimg' in os.listdir("."):
	os.system("rm -r posimg")

os.system("mkdir posimg")

fp = open("../SeaLionLoc.json", 'r')
data = json.load(fp)
fp.close()

bData = open("../SeaLionB.json", 'r')
data = json.load(fp)
fp.close()





def spotsInSub(name, subImg):
	
	for key in data[name].keys():
		if key != 'pups' and key != 'error':
			for spot in data[name][key].keys():
				nextS = data[name][key][spot]
				if nextS[0] > subImg[0][0] and nextS[1] < subImg[0][1] and nextS[1] > subImg[1][0] and nextS[1] < subImg[1][1]:

				break


def genEvenPartitions(name, img):
	imgShape = np.shape(img)
	
	for i in xrange(1, imgShape[0] // IMAGEPART, 1):
		Xshift = i * IMAGEPART
		for j in xrange(1, imgShape[1] // IMAGEPART, 1):
			Yshift = j * IMAGEPART
			spots = spotsInSub(name, [(Xshift - IMAGEPART, Xshift),(Yshift - IMAGEPART, Yshift)])
			subImg =  img[Xshift - IMAGEPART:Xshift][Yshift - IMAGEPART:Yshift]
			if spots is None:
				cv2.imwrite("./negimg/" + name + str(i) + str(j) + ".jpg", subImg)
				background.write("negimg/" + name + str(i) + str(j) +"\n")


			break
		break




for key in data.keys():

	img = cv2.imread("../Train/" + key)

	genEvenPartitions(key, img)
	break



background.close()
positive.close()
