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



def Main31(FileName):
	"""
	#==============================================================================
	#Initial
	#==============================================================================
	"""
	if DEBUG:
		print("Pre-treatment")

	img = np.array(Image.open(FileName).convert("L"))


	"""
	#==============================================================================
	#Gap Statistic and K-means
	#==============================================================================
	"""
	#==============================================================================
	#Get Data
	BlockData = []
	for i in range(0, len(img)):
		for j in range(0, len(img[i])):
			BlockData.append([float(i), float(j), float(img[i][j])])

	if DEBUG:
		print("Algorithm begin")
	#==============================================================================
	#Gap Statistic
	optimalK = OptimalK(parallel_backend = 'joblib')
	N_Cluster = optimalK(np.array(BlockData), cluster_array = np.arange(1, 50))

	if DEBUG:
		print(N_Cluster)


	#==============================================================================
	#Build K-means
	KMResult = KMeans(n_clusters = N_Cluster, random_state = 10).fit_predict(BlockData)
	if DEBUG:
		print("Output")
	TemImg = [[[0 for n in range(len(img[0]))] for n in range(len(img))] for n in range(N_Cluster)]
	for i in range(0, len(KMResult)):
		Figure = KMResult[i]
		LocX = int(BlockData[i][0] + 0.1)
		LocY = int(BlockData[i][1] + 0.1)
		TemImg[Figure][LocX][LocY] = 255

	OutImg = np.array([[0 for n in range(len(img[0]))] for n in range(len(img))])

	for i in range(0, N_Cluster):
		OutImg += np.array(cv2.Canny(np.uint8(TemImg[i]), 85, 170))
	for i in range(0, len(OutImg)):
		for j in range(0, len(OutImg[i])):
			OutImg[i][j] = 255 - min(OutImg[i][j], 255)

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
#Main("Figure/Difficult.png")
#Main("Figure/Factory.png")
#Main("Figure/Heart.png")


Main31("Figure/Inp1.jpg")
Main31("Figure/Inp3.jpg")
Main31("Figure/Inp4.jpg")
Main31("Figure/Inp5.jpg")
Main31("Figure/Inp7.jpg")
Main31("Figure/Inp8.jpg")
Main31("Figure/Inp9.jpg")







#Azunya
#445 102 414 25 37 36 200 48 155 184 261 327 4 256 238 196 160 305

#Factory Image
#43 36 96 76 114 117 159 169 172 59 57 154 64 42











