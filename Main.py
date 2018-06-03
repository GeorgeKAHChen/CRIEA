############################################################
#
#		Random Walk Image Edge Alorithm
#		Copyright(c) KazukiAmakawa, all right reserved.
#		Main.py
#
############################################################

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
import Pretreatment
import Algorithm
DEBUG = Constant.DEBUG

def Main(FileName):
	"""
	#==============================================================================
	#Initial
	#==============================================================================
	"""
	if DEBUG:
		print("Pretreatment")
	img = np.array(Image.open(FileName).convert("L"))



	"""
	#==============================================================================
	#Toboggan Algorithm
	#==============================================================================
	"""
	TobImg, TobBlock = Algorithm.Toboggan(img)




	"""
	#==============================================================================
	#Get the seed pixels
	#==============================================================================
	"""
	Seed = Algorithm.GetSeed(img)
	TobSeed = []
	for i in range(0, len(Seed)):
		TobBlock[ TobImg [Seed[i][0]] [Seed[i][0]] ] [0] = 0
		TobSeed.append(TobImg[Seed[i][0]][Seed[i][1]])
	if len(TobBlock) == len(TobSeed) + 1:
		print("Special Loop")
		for i in range(0, len(TobBlock)):
			TobBlock[i][0] = i
		OutImg = Algorithm.TobBoundary(TobImg, TobBlock, len(TobSeed))
		OutImg = Pretreatment.CombineFigures(img, OutImg, 1)
		misc.imsave("Saving/result" + str(Init.GetTime()) + ".png", OutImg)	
		return

	"""
	#==============================================================================
	#Build Probability Matirx
	#==============================================================================
	"""
	if DEBUG:
		print("Main Calculation")
	ProbArr = Algorithm.ProbCal(TobBlock, TobSeed)



	"""
	#==============================================================================
	#Decision
	#==============================================================================
	"""
	if DEBUG:
		print("Decision Part")

	#==============================================================================
	#Block Decision
	TobBlock = Algorithm.Decision(TobBlock, ProbArr)

	
	#==============================================================================
	#Output Image print
	OutImg = Algorithm.TobBoundary(TobImg, TobBlock, len(TobSeed))



	"""
	#==============================================================================
	#Decision
	#==============================================================================
	"""
	OutImg = Pretreatment.CombineFigures(img, OutImg, 1)
	misc.imsave("Saving/result" + str(Init.GetTime()) + ".png", OutImg)	




def Main3(FileName):
	"""
	#==============================================================================
	#Initial
	#==============================================================================
	"""
	if DEBUG:
		print("Pretreatment")

	img = np.array(Image.open(FileName).convert("L"))



	"""
	#==============================================================================
	#Toboggan Algorithm
	#==============================================================================
	"""
	TobImg, TobBlock = Algorithm.Toboggan(img)

	if len(TobBlock) <= 3:
		for i in range(0, len(TobBlock)):
			TobBlock[i][0] = i
		OutImg = Algorithm.TobBoundary(TobImg, TobBlock, len(TobBlock))
		OutImg = Pretreatment.CombineFigures(img, OutImg, 1)
		misc.imsave("Saving/result" + str(Init.GetTime()) + ".png", OutImg)	
		return
	


	"""
	#==============================================================================
	#Gap Statistic and K-means
	#==============================================================================
	"""
	#==============================================================================
	#Get Data
	BlockData = [[0, 0, 0] for n in range(len(TobBlock) - 1)]
	for i in range(1, len(TobBlock)):
		BlockData[i - 1][0] = TobBlock[i][1]
		BlockData[i - 1][1] = TobBlock[i][2]
		BlockData[i - 1][2] = TobBlock[i][3]



	#==============================================================================
	#Gap Statistic
	optimalK = OptimalK(parallel_backend = 'joblib')
	N_Cluster = optimalK(np.array(BlockData), cluster_array = np.arange(1, 50))

	if DEBUG:
		print(N_Cluster)


	#==============================================================================
	#Build K-means
	KMResult = KMeans(n_clusters = N_Cluster, random_state = 10).fit_predict(BlockData)
	
	#==============================================================================
	#Get Result Data
	for i in range(1, len(TobBlock)):
		TobBlock[i][0] = KMResult[i-1]
	
	#==============================================================================
	#Segmentation build
	OutImg = Algorithm.TobBoundary(TobImg, TobBlock, N_Cluster)



	"""
	#==============================================================================
	#Decision
	#==============================================================================
	"""
	OutImg = Pretreatment.CombineFigures(img, OutImg, 1)
	misc.imsave("Saving/result" + str(Init.GetTime()) + ".png", OutImg)	




def Main4(FileName):
	"""
	#==============================================================================
	#Initial
	#==============================================================================
	"""
	if DEBUG:
		print("Pretreatment")

	img = np.array(Image.open(FileName).convert("L"))



	"""
	#==============================================================================
	#Toboggan Algorithm
	#==============================================================================
	"""
	TobImg, TobBlock = Algorithm.Toboggan(img)

	print("sb")
	OutImg = [[255 for n in range(len(img[1]))] for n in range(len(img))]
	for p in range(0, len(OutImg)):
		for q in range(0, len(OutImg[p])):
			if TobImg[p][q] % 20 == 0:
				OutImg[p][q] = 0




	"""
	#==============================================================================
	#Decision
	#==============================================================================
	"""
	OutImg = Pretreatment.CombineFigures(img, OutImg, 1)
	misc.imsave("Saving/result" + str(Init.GetTime()) + ".png", OutImg)	




#Main("Figure/Easy.png")
#Main("Figure/Aznyan.png")
#Main("Figure/Caman.bmp")
Main("Figure/Difficult.png")
#Main("Figure/Factory.png")
#Main("Figure/Heart.png")


#Main("Figure/123.jpg")






#Azunya
#445 102 414 25 37 36 200 48 155 184 261 327 4 256 238 196 160 305

#Factory Image
#43 36 96 76 114 117 159 169 172 59 57 154 64 42



#Easy
#Aznyan
#Caman
#Difficult
#Factory
#Heart
"""
1402 238 958 743
455 126 411 25 426 66 42 14 179 66 40 280 251 341 353 292 379 271 203 262 200 327 
45 47 225 60 225 213 11 204 141 226 94 213 67 150 48 229 161 160 122 66 216 157 148 159
347 268 533 245 309 742 533 528 872 925 380 240
33 12 

"""

"""
26 401 546 407 501 126 130 159 321 52 305 15 303 291 154 566 478 573 366 467 324 488 26 524 603 505 
"""












