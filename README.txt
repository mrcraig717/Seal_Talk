I build the directory structure like so.............
./Train ########Is where the Train Images have to go I didn't want to put a thousand flags for which path to find them
./TrainDotted ############Train dotted Images
./TestTrain  ######## I but this directory here so you can pull off some of the Train Images for validation
./TestTrainDotted ########## Same images as in TestTrain but the Dotted version.
./Test  ####### Put the actual Test images from Kaggle here
./TrainBoxed  #### This folder was put here so we would have some where to dump images 
		   if you want to visually inspect what the bounding box procedure returned 
		   See the TestScript.py at the very bottom shows how to generate these
./GenTrainImages #######This is where I put all the code that is called from TestScript.py Ideally Craig will deal with it there is no comments in it
			See TestScript.py for instruction on untilizing what I wrote lots of comments and instructions there

The TestScript and its functions will drop json and pickle files in the main directory so when you run each stage as laid out
in the comments of the TestScript.py file you will find new files don't delete them unless you want to run it all over again

Stages for generating final JSON with bounding Boxes
Stage1: Locate and Classifiy all the spots. This ends up taking much more time then any other stage.
		###We should only have to run "hopefully" once then there will be a json file dropped in the directory
		   and we won't have to run it again

Stage1.1: I wrote something that will give some measure on how well the BlobDetector worked arcoss the whole training set see TestScript.py Comments

			Note: on the whole 900 some images this is gonna take about 4 hours.
				I currently have the parreleization at 4 processes depending on the server len we might be
				able to bump this to 8. If its a windows based system I suggest passing parellel=False
				last I checked the python multiprocessing package doesn't work on Windows. (It was a couple years ago.)

Stage2: Gen Color Classifier
	See TestScript for more info.... I think we should use the existing in SeaLoinCC.pickle first then go from there

Stage3: Gen the final bounding Boxes.
	See TestScript Comments


Extra Utilitites.... See bottom of Testscript.py
 



