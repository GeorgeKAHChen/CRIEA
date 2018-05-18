############################################################
#
#		Random Walk Image Edge Alorithm
#		Copyright(c) KazukiAmakawa, all right reserved.
#		Algorithm.py
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

#import Files
import Constant


def GetSeed(img):
	Seed = []
	while 1:
		#==============================================================================
		#Image print
		print("Now, input the seed points")
		plt.imshow(img, cmap="gray")
		plt.axis("off")
		plt.show()


		#==============================================================================
		#Seed input
		tem = input()
		tem += " "

		#==============================================================================
		#Data Initial
		Fail = False
		PreSeed = []
		ValStr = ""

		#==============================================================================
		#Str analysis
		for i in range(0, len(tem) ):
			if ord(tem[i]) < ord("0") or ord(tem[i]) > ord("9") :
				if len(ValStr) != 0:
					try:
						PreSeed.append(int(ValStr))
					except ValueError:
						print("Input Error")
						Fail = True
						break
				ValStr = ""
			else:
				ValStr += tem[i]

		#==============================================================================
		#Total number check
		if len(PreSeed) % 2 == 1 and Fail == False:
			print("Input Error")
			Fail = True

		#==============================================================================
		#Segmentation check
		if Fail == False:
			for i in range(0, int(len(PreSeed) / 2 + 0.1)):
				Seed.append([PreSeed[2 * i + 1], PreSeed[2 * i]])
				try:
					sb = img [PreSeed[2 * i + 1]] [PreSeed[2 * i]] 
				except:
					Seed = []
					print("Location out of range")
					Fail = True
					break
					
		#==============================================================================
		#Fail judgement
		if Fail:
			continue
		else:
			break

	return Seed




def Toboggan(img):
	SavArr = [[-1 for n in range(len(img[0]))] for n in range(len(img))]
	Gradient = [[0 for n in range(len(img[0]))] for n in range(len(img))]

	#Get Gradient
	img1 = cv2.Sobel(img, cv2.CV_16S, 1, 0)
	img2 = cv2.Sobel(img, cv2.CV_16S, 0, 1)

	for i in range(0, len(img1)):
		for j in range(0, len(img1[i])):
			Gradient[i][j] = math.sqrt(pow(img1[i][j], 2)+pow(img2[i][j], 2))

	Tem = 0
	Tem1 = -1
	Color = [[0, 0]]
	Loc = [[0, 0]]
	#MainLoop
	for i in range(0, len(SavArr)):
		for j in range(0, len(SavArr[i])):
			if SavArr[i][j] != -1:
				continue

			Stack = [[i, j]]
			Tem += 1
			Color.append([0, 0])
			Loc.append([0, 0])
			while 1:
				if len(Stack) == 0:
					break

				Block = []
				Vari = Stack[len(Stack)-1][0]
				Varj = Stack[len(Stack)-1][1]
				Stack.pop()
				if SavArr[Vari][Varj] == -1:
					SavArr[Vari][Varj] = Tem
					Color[len(Color)-1][0] += 1
					Color[len(Color)-1][1] += img[Vari][Varj]
					Loc[len(Color)-1][0] += Vari
					Loc[len(Color)-1][1] += Varj
				else:
					continue
			
				if Tem != Tem1:
					print("Block:\t" + str(Tem), end = "\r")
					Tem1 = Tem

				for p in range(-1, 2):
					for q in range(-1, 2):
						Poi = 0
						try:
							Poi = Gradient[Vari+p][Varj+q]
							Block.append([Gradient[Vari+p][Varj+q], Vari+p, Varj+q])
						except:
							continue
						

				Block.sort()
				for k in range(0, len(Block)):
					if SavArr[Block[k][1]][Block[k][2]] == -1 and Block[k][1] != Loc[len(Color)-1][0] and Block[k][2] != Loc[len(Color)-1][1]:
						#This judgement may have some bug
						Stack.append([Block[k][1], Block[k][2]])
						break
					
	print("Block:\t" + str(Tem), end = "\n")
	BlockInfo = [[0, 0, 0, 0, 0]]

	for i in range(1, len(Color)):
		Tem = [-1]
		Tem.append(abs(int(Color[i][1]/Color[i][0])))
		Tem.append(abs(int(Loc[i][0]/Color[i][0])))
		Tem.append(abs(int(Loc[i][1]/Color[i][0])))
		Tem.append(0)
		BlockInfo.append(Tem)

	for i in range(0, len(SavArr)):
		for j in range(0, len(SavArr[i])):
			if SavArr[i][j] == -1:
				continue
			BlockInfo[SavArr[i][j]][4] += 1

	return [SavArr, BlockInfo]



def ProbCal(TobBlock, TobSeed):
	Varu = len(TobBlock) - len(TobSeed) - 1
	Varl = len(TobSeed)
	Puu = []
	Pul = []

	for i in range(0, len(TobBlock)):
		if TobBlock[i][0] != -1:
			continue
		PuuLine = []
		PulLine = []
		TTL = 0
		for j in range(0, len(TobBlock)):
			weight = math.exp(- Constant.alpha * (TobBlock[i][1] - TobBlock[j][1]) - Constant.beta * ((TobBlock[i][2] - TobBlock[j][2]) ** 2 + (TobBlock[i][3] - TobBlock[j][3]) ** 2)     )
			TTL += weight
			if TobBlock[i][0] == -1:
				PuuLine.append(weight)
			else:
				PulLine.append(weight)

		for p in range(0, len(PuuLine)):
			PuuLine[p] /= TTL

		for p in range(0, len(PulLine)):
			PulLine[p] /= TTL

		Puu.append(PuuLine)
		Pul.append(PulLine)
	
	if Constant.DEBUG:
		print("Probability matrix build succeed. Decision matrix building start")
	return np.linalg.pinv(np.eye(Varu) - np.matrix(Puu)) * np.matrix(Pul)



def Decision(TobBlock, ProbArr):
	tem = 0
	for i in range(0, len(TobBlock)):
		if TobBlock[i][0] == -1:
			MaxLoc = 0
			MaxVal = 0
			for j in range(0, len(ProbArr[tem])):
				if ProbArr[tem][j] > MaxVal:
					MaxVal = ProbArr[tem][j]
					MaxLoc = j
			TobBlock[i][0] = MaxLoc
		else:
			TobBlock[i][0] = tem
			tem += 1

	return TobBlock



def TobBoundary(TobImage, TobBlock, BlockArea):
	for i in range(0, len(TobImage)):
		for j in range(0, len(TobImage[i])):
			if TobImage[i][j] == -1:
				continue

			TobImage[i][j] = TobBlock[TobImage[i][j]][0]
	
	OutImg = [[255 for n in range(len(TobImage[1]))] for i in range(len(TobImage))]
	for i in range(0, BlockArea):
		BlankImg = [[0 for n in range(len(TobImage[1]))] for i in range(len(TobImage))]
		for p in range(0, len(TobImage)):
			for q in range(0, len(TobImage[p])):
				if TobImage[p][q] == i:
					BlankImg[p][q] = 255

		BlankImg = cv2.Canny(np.uint8(BlankImg), 85, 170)
		for p in range(0, len(BlankImg)):
			for q in range(0, len(BlankImg[p])):
				if BlankImg[p][q] > 200:
					OutImg[p][q] = 0

	return OutImg






