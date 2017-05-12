from GenTrainImages import Utilities
import pickle
import cv2
import os
import sys
sys.path.append("./HaarClass/")

if len(os.listdir("./Train/")) < 2:
	print("There are no Images in the MaskPlay Train Folder If the the TrainDotted Images are not there this thing is gonna crash and burn: I not writing all the flags put the images in the folders as explained in the README")
	exit()

####GenSpots JSON
########################################################################################################
##STAGE 1: Find all the dotts in the TrainDotted Images and classifiy them

## this function will process all images that exist in the Train folder procuding a json file 
#		whith the locations of the Dotts to be Used later
# if reset is set to True it will process all images in the Train Folder erasing the current json if there is one
# !!!!!!!!!!!!!!!1This is the slowest part of the whole thing so once run once we shouldn't have to do it again
# if reset is set to False it will only process images that don't exist in the existing json File

#######################################################################################################
#Uncomment below to genSpots
Utilities.genSpots(reset=True, parellel=False)

#######################################################################################################
##I beleive the parameter in there current setting will work well for all Images but this Function will
## generate the difference between the data in train.csv file and the current SeaLionLoc.json file
## e.g. abs([100,2,35,9,0] - [99,1,34,10]) summed of all rows (number of Dots it failed to classifiy, or 
##	                                                           classified incorrectly ) 
## Running over the first small set of Images gave a 4 so we should be hoping for something less then
## 800 would be pretty good when running the entire Training Set. A number much higher then that and we should pause
## and adjust some stuff before moving on.
#######################################################################################################
#Uncomment to check location json
Utilities.checkLocJson()


#######################################################################################################
##STAGE 2:
##Generate the JSON FILE like the one posted on SEAL_TALK SLACK will run all images in the 
#Location json file 

#######################################################################################################
##Params:
#   parellel: True, False if you want it so spawn off 4 processes we can change this to 8 if the server can handle it
#	cropParam = int  Set the bound on the Max size of the boxes generated 60 would max the box size at 121 X 121 which
#                           is much larger then any Sea Lion in the Training Images
#   colorClassifier: If none uses color gradient (doesn't work well with our images), else color classifier 
#							must be an object with predict function returning 1.0, or 0.0
#                           WE can change this but thats what I have been doing so far to generate the bounding Boxes  
####################################################################################################

#Uncomment to gen json with all the bounding Boxes
Utilities.genjson(parellel=False, cropParam=60, colorClassifier=None, include=["adult_males", "subadult_males", "adult_females", "juveniles"])

# Will draw bounding boxes on images in the TrainDotted folder and place them in the TrainBoxed folder
Utilities.genALLBoxedImages()
###############################################################################3


##################################################################################################
##Load the last Haar Classifier I trained and run prediction on the files in the Train folder
###################################################################################

os.system("python HaarTrainScript.py")
os.system("python HaarPredict.py")
