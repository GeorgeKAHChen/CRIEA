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
import Init



def GetSeed(img):
	Seed = []
	while 1:
		#==============================================================================
		#Image print
		print("Now, input the seed points")
		#plt.imshow(img, cmap="gray")
		#plt.axis("off")
		#plt.show()


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

	Label = 0
	TobBlock = [[-1, 0, 0, 0, 0]]
	#MainLoop
	for i in range(0, len(SavArr)):
		for j in range(0, len(SavArr[i])):
			print("TobBlock = " + str(Label), end = "\r")
			if SavArr[i][j] != -1:
				continue

			Stack = [[i, j]]
			ImaLabel = 0
			while 1:
				Neigh = []
				for p in range(-1, 2):
					for q in range(-1, 2):
						if p == 0 and q == 0:
							continue
						LocX = i + p
						LocY = j + q
						if LocX < 0 or LocY < 0 or LocX > len(SavArr) - 1 or LocY > len(SavArr[0]) - 1:
							continue
						else:
							Neigh.append([Gradient[LocX][LocY], LocX, LocY])



				Neigh.sort()
				#print(Neigh)
				if SavArr[Neigh[0][1]][Neigh[0][2]] != -1:
					ImaLabel = SavArr[Neigh[0][1]][Neigh[0][2]]
					break

				if Neigh[0][1] != Stack[len(Stack) - 1][0] or Neigh[0][2] != Stack[len(Stack) - 1][1]:
					Stack.append([Neigh[0][1], Neigh[0][2]])
				else:
					TobBlock.append([-1, 0, 0, 0, 0])
					Label += 1
					ImaLabel = Label
					break


			while len(Stack) != 0:
				LocX = Stack[len(Stack) - 1][0]
				LocY = Stack[len(Stack) - 1][1]
				Grey = img[LocX][LocY]
				Stack.pop()
				SavArr[LocX][LocY] = ImaLabel
				TobBlock[ImaLabel][1] += Grey
				TobBlock[ImaLabel][2] += LocX
				TobBlock[ImaLabel][3] += LocY
				TobBlock[ImaLabel][4] += 1
	
	#Init.ArrOutput(TobBlock)
	print("TobBlock = " + str(Label), end = "\n")
	
	for i in range(1, len(TobBlock)):
		TobBlock[i][1] /= TobBlock[i][4]
		TobBlock[i][2] /= TobBlock[i][4]
		TobBlock[i][3] /= TobBlock[i][4]


	return [SavArr, TobBlock]



def ProbCal(TobBlock, TobSeed):
	Varu = len(TobBlock) - len(TobSeed) - 1
	Varl = len(TobSeed)
	Puu = []
	Pul = []

	for i in range(1, len(TobBlock)):
		if TobBlock[i][0] != -1:
			continue
		PuuLine = []
		PulLine = []
		TTL = 0
		for j in range(1, len(TobBlock)):
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






