import os
import json
import cv2
import numpy as np
import time
import sys
import pickle
import time
from HaarPreProc import HaarPreProc
sys.path.append("../")
from GenTrainImages import ColorMaskGenerator

##########################################################################################
#Script used for train the Haar Classifiers a lot of it commented out and was not 
# being used at the end when I was trying to get this simpilist version of it working
#########################################################################################33
CMG = ColorMaskGenerator.CMG(None)



fp = open("../SeaLionB.json", 'r')
BBdata = json.load(fp)
fp.close()

classes = ["adult_males", "adult_females", "subadult_males", "juveniles"]
#####################3
########## This section is what is used to generate the files needed for the Haar Classifier
#HPP = HaarPreProc("../Train/", BBdata, CMG, classes, 1000)
#nPOSSamples, nNEGSamples = HPP.run()# intenseMean, intenseSTD = HPP.run()


#fp = open("currentSampleDist.txt", 'w')
#fp.write("Pos Sample Dist:  " + str(nPOSSamples) + "\n")
#fp.write("Neg Sample Dist:  " + str(nNEGSamples) + "\n")
#fp.write("Sample Mean Intensity: " + str(intenseMean) + "\n")
#fp.write("Sample STD Intensity: " + str(intenseSTD))
#fp.close()
#os.system("opencv_createsamples -vec adult_males.vec -bg bg.txt -info adult_males.txt -bgthresh 80 -num 4989 -h 40 -w 40")
#for classtype in classes:
#    if nPOSSamples[classes.index(classtype)] != 0:
#        os.system("opencv_createsamples -vec " + classtype + ".vec -bg bg.txt -info " + classtype + ".txt -bgthresh 10 -num " + str(nPOSSamples[classes.index(classtype)]) + " -h 40 -w 40")
#####################################33



#if 'AMcascade' in os.listdir("."):
#    os.system("rm -r AMcascade")
#os.system("mkdir AMcascade")
# if 'AFcascade' in os.listdir("."):
# 	os.system("rm -r AFcascade")
# os.system("mkdir AFcascade")
#if 'SAMcascade' in os.listdir("."):
#   os.system("rm -r SAMcascade")
#os.system("mkdir SAMcascade")
# if 'Jcascade' in os.listdir("."):
#     os.system("rm -r Jcascade")
# os.system("mkdir Jcascade")

# ####### adult_male Train Command
os.system("opencv_traincascade -data AMcascade -vec adult_males.vec -bg bg.txt -numStages 20 -minHitRate .92 -maxFalseAlarmRate .5 -numPos 249 -numNeg 300 -maxDepth 1 -h 40 -w 40")
# ####### adult_female Train Command
#os.system("opencv_traincascade -data AFcascade -vec adult_females.vec -bg bg.txt -numStages 6 -minHitRate .90 -maxFalseAlarmRate .4 -featuretype LBP -numPos 100 -numNeg 150 -h 40 -w 40")
# #######subadult_males Train Command
#os.system("opencv_traincascade -data SAMcascade -vec subadult_males.vec -bg bg.txt -numStages 6 -minHitRate .99 -maxFalseAlarmRate .5 -numPos 200 -numNeg 200 -h 24 -w 36")
# ######Juvinile Train Command
# os.system("opencv_traincascade -data Jcascade -vec juveniles.vec -bg bg.txt -numStages 6 -minHitRate .90 -maxFalseAlarmRate .4 -numPos 110 -numNeg 150 -h 40 -w 40")