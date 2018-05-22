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

def Main():
	"""
	#==============================================================================
	#Initial
	#==============================================================================
	"""
	if DEBUG:
		print("Pretreatment")

	img = np.array(Image.open(Constant.ImageName).convert("L"))



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



	"""
	#==============================================================================
	#Build Probability Matirx
	#==============================================================================
	"""
	if DEBUG:
		print("Main Calculation")
	ProbArr = Algorithm.ProbCal(TobBlock, TobSeed)
	if DEBUG:
		pass
		#Init.ArrOutput(ProbArr)



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
	misc.imsave("Saving/result.png", OutImg)	




def Main3():
	"""
	#==============================================================================
	#Initial
	#==============================================================================
	"""
	if DEBUG:
		print("Pretreatment")

	img = np.array(Image.open(Constant.ImageName).convert("L"))



	"""
	#==============================================================================
	#Toboggan Algorithm
	#==============================================================================
	"""
	TobImg, TobBlock = Algorithm.Toboggan(img)

	

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
	misc.imsave("Saving/result.png", OutImg)	







Main3()







#Azunya
#445 102 414 25 37 36 200 48 155 184 261 327 4 256 238 196 160 305

#Factory Image
#43 36 96 76 114 117 159 169 172 59 57 154 64 42











