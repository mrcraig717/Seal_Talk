import cv2
import json
import os
import numpy as np


class GenGoogleNetPart:
    def __init__(self, size=(384, 1248)):
        self.size = size
        if "TrainP" in os.listdir("."):
            os.system("rm ./TrainP/*.jpg")
            
        else:
            return None
        fp = open("SeaLionB.json", "r")
        self.data = json.load(fp)
        fp.close()
        self.newJson = {}
    
    def run(self):

        for name in self.data.keys():
            print("Generating Partitions for Img: " + name)
            onPartition = 0
            img = cv2.imread("./Train/" + name)
            imgShape = np.shape(img)
            for i in xrange(1, imgShape[0] // self.size[0], 1):
                Xshift = i * self.size[0]
                for j in xrange(1, imgShape[1] // self.size[1], 1):
                    Yshift = j * self.size[1]
                    subImg =  img[Xshift - self.size[0]:Xshift, Yshift - self.size[1]:Yshift].copy()
                    cv2.imwrite("./TrainP/" + name.split(".")[0] + "_" + str(onPartition) + ".jpg", subImg)
                    self.newJson[name.split(".")[0] + "_" + str(onPartition) + ".jpg"] = self.spotsInSub(name, [(Xshift - self.size[0], Xshift),(Yshift - self.size[1], Yshift)])
                    onPartition += 1
        fp = open("./TrainP/SeaLionB.json" , 'w')
        json.dump(self.newJson, fp)
        fp.close()

    def spotsInSub(self, name, subImg):
        result = {}
        for key in self.data[name].keys():
            result[key] = {}
            if key != 'pups' and key != 'error':
                for spot in self.data[name][key].keys():
                    temp = spot.split(",")
                    nextS = (int(temp[0][1:]), int(temp[1][:-1]))
                    ####Check to see if the spot is inside our subImg
                    if nextS[0] > subImg[0][0] and nextS[0] < subImg[0][1] and nextS[1] > subImg[1][0] and nextS[1] < subImg[1][1]:
                        result[key][spot] = self.getBBCrop(self.data[name][key][spot], subImg)
        return result

    def getBBCrop(self, BB, subI):
    	if BB[1][0] < subI[0][0]:
        	Xmin = 0
    	else:
        	Xmin = BB[1][0] - subI[0][0] - 1
    
    	if BB[1][1] < subI[1][0]:
        	Ymin = 0
    	else:
        	Ymin = BB[1][1] - subI[1][0] - 1
    
    
    	if BB[0][0] > subI[0][1]:
        	Xmax = self.size[0] - 1
    	else:
        	Xmax = (BB[0][0] - subI[0][0])
    
    	if BB[0][1] > subI[1][1]:
        	Ymax = self.size[1] - 1
    	else:
			Ymax = (BB[0][1] - subI[1][0])
    
    	return [(Xmax, Ymax), (Xmin, Ymin)]
