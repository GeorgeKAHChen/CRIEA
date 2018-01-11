#=================================================================
#
#		CNN and RW Image Edge Algorithm
#		Copyright(c) KazukiAmakawa, all right reserved.
#		Pretreatment.py
#
#=================================================================
"""
		FUNCTION INSTRUCTION

FigureInput(Model)
	This function will return the figure you have.
	NOTE: THIS FUNCTION WILL NOT READ THE FUGURE AT "Output/"
	
	Model = The work model you need, 1 means read files except "Output/"
									 2 means read files only "Output/"
	
	return Figure Location Array

Output(img, Name, kind)
	This fucntion will save files as the name input
	NOTE: ALL FILES WILL BE SAVED UNDER THE FILTER "Output/" and "Saving/"
	NOTE: EVERY TIME YOU BEGIN THIS SYSTEM, ALL FILES IN "Output/" WILL BE DELETED.

	img = The image array you want to save
	Name = The name of the array you want to save 
	kind = 1: It is temporary saving into "Output/"
		   2: The file will be saved in "Saving/" for long time.
	return None

FigurePrint(img, kind)
	This function will print the figure with matplotlib tools

	img = The image you want to print
	kind = 1: You can save the figure after print.
		   2: You can just see the figure.
		   3: The function will save figure auto.

	return None

FigureSave(img, FileName)
	This function will save img at /Output

	img = image Array you worked on it
	FileName = The name of the file

	return None

BFSmooth(img)
	This function will return fugure you need smooth

	img  = image Array you worked on it

	return Image Array after smoothing


Partial(img)
	This function will cut the figure and save them as small blocks

	img = The image array you want cut
	Global variable:  FigSize: The size of figure will be cut

	return The number of block had been cut
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
from collections import deque
from PIL import ImageFilter
import cv2
from copy import deepcopy

#import files
import Init
import Functions


#Constant
FigSize = 300
#The size of figure will be cut


def FigureInput(model):
	Figure = []
	Name = []
	root0 = ""
	for root, dirs, files in os.walk(os.getcwd()):
		if root0 == "":
			root0 = root
			if Init.SystemJudge() == 0:
				root0 += "/Saving"
			else:
				root0 += "\\Saving"
		if model == 1 and root0 == root:
			continue
		if model == 2 and root0 != root:
			continue

		for i in range(0, len(files)):
			if Init.SystemJudge() == 0:
				LocStr = root + "/" + files[i]
			else:
				LocStr = root + "\\" + files[i]
			Hajimari = 0
			Last = ""

			for j0 in range(0, len(files[i])):
				j = len(files[i]) - j0 -1
				if files[i][j] == ".":
					break
				else:
					Last = files[i][j] + Last

			if Last == "bmp" or Last == "jpg" or Last == "png":
				Figure.append(LocStr)
				Name.append(files[i])

	if len(Figure) == 0:
		print("No Figure")
		return -1

	print("FileLise: ")
	for i in range(0, len(Figure)):
		print(str(i+1) + "\t" + Name[i])
	
	Init.LogWrite("Initialization succeed","0")
	return Figure


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


def BFSmooth(img):
	Init.LogWrite("Bilateral Filter Smooth succeed","0")
	return cv2.bilateralFilter(img, 9, 40, 40)


def Partial(img):
	BlockY = (len(img) - 1) // FigSize
	BlockX = (len(img[0]) - 1) // FigSize
	Tem = 0
	BlockInfo = [[0, 0, 0, 0]]
	for i in range(0, BlockY + 1):
		for j in range(0, BlockX + 1):
			Tem += 1
			SaveImg = []
			p = 0
			q = 0
			for p in range(0, FigSize + 1):
				Line = []
				for q in range(0, FigSize + 1):	
					try:
						Line.append(img[i * FigSize + p][j * FigSize + q])
					except:
						break
				if len(Line) != 0:
					SaveImg.append(Line)
				else:
					break
			#print([i, j, p, q])
			#print(img)
			BlockInfo.append([i, j, len(SaveImg), len(SaveImg[0])])
			Output(SaveImg, "Block_" + str(Tem) + ".png", 1)

	BlockInfo[0][2] = FigSize * BlockX  + BlockInfo[len(BlockInfo)-1][2]
	BlockInfo[0][3] = FigSize * BlockY  + BlockInfo[len(BlockInfo)-1][3]
	#print(BlockInfo[len(BlockInfo)-1][2], BlockInfo[len(BlockInfo)-1][3])
	return BlockInfo


def Recovery(BlockSize, BlockInfo):
	img = [[0 for n in range(BlockInfo[0][3] + 1)] for n in range(BlockInfo[0][2] + 1)]
	for kase in range(1, len(BlockInfo)):
		img1 = np.array(Image.open("Output/Block_" + str(kase) + ".png").convert("L"))
		for i in range(0, len(img1)):
			for j in range(0, len(img1[i])): 
				img[FigSize * BlockInfo[kase][0] + i][FigSize * BlockInfo[kase][1] + j] = img1[i][j]
	FigurePrint(img, 3)


