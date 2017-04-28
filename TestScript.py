from GenTrainImages import Utilities
import pickle
import cv2
import os

####GenSpots JSON
##########################################################################################33
##STAGE 1: Find all the dotts in the TrainDotted Images and classifiy them

## this function will process all images that exist in the Train folder procuding a json file whith the locations of the Dotts to be Used later
# if reset is set to True it will process all images in the Train Folder erasing the current json if there is one
# !!!!!!!!!!!!!!!1This is the slowest part of the whole thing so once run once we shouldn't have to do it again
# if reset is set to False it will only process images that don't exist in the existing json File

#######################################################################################################
#Uncomment below to genSpots
#Utilities.genSpots(reset=True, parellel=False)

############################################################################################
##I beleive the parameter in there current setting will work well for all Images but this Function will
## generate the difference between the data in train.csv file and the current SeaLionLoc.json file
## e.g. abs([100,2,35,9,0] - [99,1,34,10]) summed of all rows (number of Dots it failed to classifiy, or 
##	                                                           classified incorrectly ) 
## Running over the first small set of Images gave a 4 so we should be hoping for something less then
## 800 would be pretty good when running the entire Training Set. A number much higher then that and we should pause
## and adjust some stuff before moving on.
################################################################################################
#Uncomment to check location json
#Utilities.checkLocJson()

###############################################################################################################
##STAGE 2: Generate Color Classifier I have set it to use Logitsic Regression and can change pretty easy to anything esle 

## I think for first we should use the color classifier that I already trained 
################################################################################################################

####GenColorClassifier
## Parellel is True is colllects colors from the pictures in parellel spawns four Processes as of Now
## PARMS:
#	cropParam: how many pixels to extend from the location of the Dot
#	sizeParam: max number of crops to make from each image spreads them over negative and positive samples
#	parellel: True Spawns 4 Process to do the croping procedure

#I left this commented out for now I think we just the one I trained and is in the main project folder first 
#              This takes a long time depending on params set total number of data points collected:
#                         < numOfTrainImages * (2 * sizeParam *(2 * cropParam + 1)^2) 

##############################################################################################################


##Uncomment to generate new classifier
#Utilities.gencolorclassifier(parellel=False, cropParam=1, sizeParam=10)

###############################################################################################################
# Load the Color Classifier
if "SeaLionCC.pickle" in os.listdir("."):
	fp = open("SeaLionCC.pickle", 'r')
	CC = pickle.load(fp)
	fp.close()
else:
	print("No color Classifier pickle in working directory")


#####################################################################################################
##STAGE 3:
##Generate the JSON FILE like the one posted on SEAL_TALK SLACK will run all images in the 
#Location json file 

#####################################################################################################
##Params:
#   parellel: True, False if you want it so spawn off 4 processes we can change this to 8 if the server can handle it
#	cropParam = int  Set the bound on the Max size of the boxes generated 60 would max the box size at 121 X 121 which
#                           is much larger then any Sea Lion in the Training Images
#   colorClassifier: If none uses color gradient (doesn't work well with our images), else color classifier 
#							must be an object with predict function returning 1.0, or 0.0
#                           WE can change this but thats what I have been doing so far to generate the bounding Boxes  
####################################################################################################

#Uncomment to gen json with all the boundi
Utilities.genjson(parellel=False, cropParam=60, colorClassifier=CC)


###############################################################################3
# Extra Utilites:

###Will Generate a image with the bounding boxes found in current SeaLionB.json file and dump it into
##   ./Train/TrainBoxed/ directory

# Params File name

Utilities.genBoxedImage("44.jpg")


