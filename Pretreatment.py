############################################################
#
#		Random Walk Image Edge Alorithm
#		Copyright(c) KazukiAmakawa, all right reserved.
#		Pretreatment.py
#
############################################################
"""
		FUNCTION INSTRUCTION
		
This is the main function of all the program.

Histogram(img)
	This function will statistic the probability of all grey level
	
	img = Array of the figure 
	
	return Histogram array based on probability 


DerHis(HisArr)
	This function will get the second order derivative of the 
	histogram array

	HisArr = The histogram array of the image

	return 2nd order derivative array


Sta2Der(DerArr)
	This function will find the zero crossing of the histogram array

	DerArr = The second derivative array
	
	return Array of Pair of zero crossing as [[minvalue, maxvalur]...]


ZCAnalysis(img, GausData)
	This algorithm will get zero crossing data of the histogram

	img = [image array]
	GausData = [Histogram after gaussian smoothing]

	return image after Zero crossing 


FigurePrint(img, kind):
	This function will print and output a image
	
	img = [image array]
	kind = working method

	**This function is a historical function, do not using it any more.


ProbLearn(HisArr, ZeroC)
	This function will return fully histogram after expose
	
	HisArr = [Array of histogram]
	ZeroC = [Data of initial parameter interval]
	
	return Parameter interval after expose


Thresholding(img, TSH, TSH2)
	This algorithm will change the grey pixel from THS to THS2 as white and return the boundary

	img = [array of the whole image]
	THS = the minimum thresholding
	THS = the maxinum thresholding


Output(img, Name, kind)
	**DO NOT USE THIS FUCNTION ANY MORE


AutoTH(Histogram, varTH):
	**DO NOT USE THIE FUNCTION ANY MORE


Partial(img):
	This function will block a large image as small block

	img = [array of large image ]

	return Infomation of all blocks

	**CAUTION: This function will save all the blocked image as files



Recovery(BlockSize, BlockInfo, ImageName):
	This function will recovery all images from protial


CombineFigures(img1, img2, model):
	#This function will combine two figures in different RGB channel
	
	img1 = initial image
	img2 = boundary image
	model = 1, means background is 255 and 
		    0, means 0
	
	return the RGB boundary image


GetPeak(Histogram, N_Cluster):
	This function will get the peak of image from N_Cluster as close as possible.
	
	Histogram = [histogram array]
	N_Cluster = the number of peak you need 

	return The data of all peak interval


CARLA(Histogram, PairOfZC)
	Continuous Action Reinforcement Learning Automaton Method Main Function
	This function will use CARLA to solve a GMM model to connect the input Histogram

	Histogram = [histogram which want to likelihood]
	PairOfZC = [learning initial data]

	return Data after learning

	**CAUTION:1st TO DETERMINE CALLING C CODE SUCCEED, IT IS NECESSARY TO CALL Pre_CARLA()
				  FUNCTION BEFORE USING THE FUNCTION
			  2nd THIS FUNCTION WILL BUILD TWO FILES, ONE IS tem.py TO SAVE LEARNING DATA
			  	  THE OTHER IS Input.out SAVE INITIAL DATA FOR C CODE
			  3rd THIS FUNCTION WILL READ DATA FROM THE FILE Output.out AFTER C CODE CALLING
			  4th YOU CAN CHANGE THE LEARNING FUCNTION IN Constant.py


Pre_CARLA():
	This is the pre-treatment function of CARLA,
	
	return None

	** CAUTION: IF YOU ARE LINUX OR MACOS USER, IT IS NECESSARY TO DETERMINE THE LOCATION OF 
			    PYTHON COMPILE FILE BEFORE YOU USE THIS FUNCTION. YOU CAN DETERMINE IT IN 
			    Constabt.py 


GMM_THS(Histogram, DataLast):
	This fucntion will return GMM Ths after CARLA 

	Histogram = [histogram learning]
	DataLast = [data after learning]

	return Thresholding in GMM


Histogram(TobBlock)
	This function will return histogram after Toboggan

	TobBlock = Toboggan algorithm block infomathon

	return histogram
"""


#import head
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import math
import matplotlib.patches as patches
from scipy import misc
from scipy import signal
from collections import deque
from PIL import ImageFilter
import cv2
from copy import deepcopy
import random
import pandas as pd
from gap_statistic import OptimalK
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans



#import Files
import Init
import Constant


def FigurePrint(img, kind):
	plt.imshow(img, cmap="gray")
	plt.axis("off")
	plt.show()
	Init.LogWrite("Figure output succeed", "0")
	if kind == 1:
		InpStr = input("Save the figure?[Y/ n]")
		if InpStr == "Y" or InpStr == "y":
			Name = "Figure_"
			Name += str(Init.GetTime())
			Name += ".png"
			Output(img, Name, 2)
	elif kind == 1:
		pass
	elif kind == 3:
		Name = "Figure_"
		Name += str(Init.GetTime())
		Name += ".png"
		Output(img, Name, 2)
	return



def Output(img, Name, kind):
	if Init.SystemJudge() == 0:
		os.system("cp null " + Name)
		misc.imsave(Name, img)
		if kind == 1:
			if not os.path.exists("Output"):
				os.system("mkdir Output")
			os.system("mv " + Name + " Output/" + Name)
		if kind == 2:
			if not os.path.exists("Saving"):
				os.system("mkdir Saving")
			os.system("mv " + Name + " Saving/" + Name)
	else:
		os.system("copy null " + Name)
		misc.imsave(Name, img)
		if kind == 1:
			if not os.path.exists("Output"):
				os.system("mkdir Output")
			os.system("move " + Name + " Output/" + Name)
		if kind == 2:
			if not os.path.exists("Saving"):
				os.system("mkdir Saving")
			os.system("move " + Name + " Saving/" + Name)		
	return



def CombineFigures(img1, img2, model):
	#This function will combine two figures in different RGB channel
	#Where img1 is initial image, img2 is boundary image
	#Also, if model = 1, means background is 255 and 0 means 0
	img = []
	Judge = 0
	if model == 1:
		Judge = 255
	for i in range(0, len(img2)):
		imgLine = []
		for j in range(0, len(img2[i])):
			if img2[i][j] != Judge:
				imgLine.append([255, 0, 0])
			else:
				imgLine.append([img1[i][j], img1[i][j], img1[i][j]])
		img.append(imgLine)
	#Init.ArrOutput(img, 1)
	return np.array(img)





