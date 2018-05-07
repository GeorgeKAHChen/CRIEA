#=================================================================
#
#		CNN and RW Image Edge Algorithm
#		Copyright(c) KazukiAmakawa, all right reserved.
#		Algorithm.py
#
#=================================================================
"""
		FUNCTION INSTRUCTION
		
Toboggan(img)
	This function established the Toboggan Algorithm

	img = Image Matrix will calculate

	return [Array with sign, Point information[Block Code, Average Gray, Center Local X, Center Local Y]]

WeightFunc(Point1, Point2)
	This function will calculate the weight between two node

	Point1 = [BlockCode1, Grey1, LocX1, LocY1]
	Point2 = [BlockCode2, Grey2, LocX2, LocY2]
	
	Global variable:   Alpha: A parameter about the relationship between Grey and Distance
	Global variable:   Beta:  A parameter about the relationship between Grey and Distance

	return The probability between two node
	
GetProb(NodeInfo)
	This function will calculate the weight matrix between nodes

	NodeInfo = [number of node][BlockCode1, Grey1, LocX1, LocY1]

	return Probability Matrix between nodes

HandSeed(Tobimg, img)
	You can choose the pixels in different Toboggan blocks

	Tobimg = Toboggan Blocked array
	img = The normal imgage

	return the set of upground and background

LinFuncGroup(InpArr, VarN)
	This function will solve the linear functions groups and get the probability 
	from unsigned nodes to the seed nodes.

	InpArr = The linear functions groups of AX = B , and it is [A B] Matrix
	VarN = The number of variables need to solve

	return Probability Solution Matrix

GetProbomatrix(Block,Seedset)
	This function will get Probability Matrix Calculation

	Block = Block infromation s.t. [BlockCode, Grey, LocX, LocY]
	Seedset = Seedset

	return [[Probability Matrix,sizen,sizel],NoSeedBlock List]

GetClassifyProb(Pln)
	This function will get the Probability solution.

	Pln = [P,n,l]
	
	return Probability Matirx

RWclassify(Py,NoseedBlock)
	This function will get the solution from one block to another

	Py = Probability Matrix	
	NoseedBlock = The array which saved the Nodes not from any part of Toboggan block

	return [Part of Probability, kind, blockcode]

BlockFigure(Tobimg, ProbBlock, LenSeed)
	This function will return the figure after classify and thresholding

	Tobimg = The output img after toboggan algorithm
	ProbBlock = The result of probability matrix of all the figure
	LenSeed = The length of seed array

	return The figure after classified and thresholding

GetBoundary(Tobimg):
	This function will return the boundary array.

	Tobimg = the image which classified after BlockFigure function

	return The boundary figure

Laplacian(NodeInfo, VarL):
	This function will return two part of Laplacian matrix after block

	NodeInfo = The information of all nodes which is seed first
	VarL = The length of seed set

	return Luu Lul , two part of Laplacian Matrix in one array
		
	Note: We can show a Laplacian matrix after block as:
	LM = Lll Llu
		 Lul Luu
	As we just need the "Lul" part and "Luu" part for the caculation after,
	we will return these part of matrix.

	Note: The output is Normalized Laplacian matrix, that means, the whole 
	matrix satisfied the equal of line is zero.

SobelAlg(img):
	This function will establish the function of sobel operator

	img = the array of figure you want to get boundary

	return boundary figure array
	
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
from scipy import ndimage
from collections import deque
from PIL import ImageFilter
import cv2
from copy import deepcopy


#import files
import Init
import Pretreatment
import Functions
import Constant


def Toboggan(img):
	SavArr = [[-1 for n in range(len(img[0]))] for n in range(len(img))]
	Gradient = [[0 for n in range(len(img[0]))] for n in range(len(img))]

	#Get Gradient
	img1 = cv2.Sobel(img, cv2.CV_16S, 1, 0)
	img2 = cv2.Sobel(img, cv2.CV_16S, 0, 1)

	for i in range(0, len(img1)):
		for j in range(0, len(img1[i])):
			Gradient[i][j] = int(math.sqrt(pow(img1[i][j], 2)+pow(img2[i][j], 2)))

	Tem = 0
	Tem1 = -1
	Color = [[0, 0]]
	Loc = [[0, 0]]
	#MainLoop
	for i in range(1, len(SavArr)-1):
		for j in range(1, len(SavArr[i])-1):
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
						except:
							continue
						Block.append([Gradient[Vari+p][Varj+q], Vari+p, Varj+q])

				Block.sort()
				for k in range(0, len(Block)):
					if SavArr[Block[k][1]][Block[k][2]] == -1 and Block[k][1] > 0 and Block[k][2] > 0:
						#This judgement may have some bug
						Stack.append([Block[k][1], Block[k][2]])
						break
					
	print("Block:\t" + str(Tem), end = "\n")
	BlockInfo = [[0, 0, 0, 0]]
	#Init.ArrOutput(Loc)
	for i in range(1, len(Color)):
		Tem = [i]
		Tem.append(abs(int(Color[i][1]/Color[i][0])))
		Tem.append(abs(int(Loc[i][0]/Color[i][0])))
		Tem.append(abs(int(Loc[i][1]/Color[i][0])))
		BlockInfo.append(Tem)

	Init.LogWrite("Toboggan Algorithm run succeed","0")

	return [SavArr, BlockInfo]
"""
I am not sure, but I think this toboggan still have some bug which not confident
I deal with this bug directly, which I take all output as abs().
I signed the line which maybe have some bug.
"""


def WeightFunc(Point1, Point2):
	Grey = math.sqrt(pow( (Point1[1] - Point2[1]) , 2))
	X = pow( (Point1[2]-Point2[2]) , 2)
	Y = pow( (Point1[3]-Point2[3]) , 2)
	Distance = math.sqrt(X + Y)
	return math.exp( - Constant.Alpha * Grey - Constant.Beta * Distance )


def GetProb(NodeInfo):
	Prob = [[0.00 for n in range(len(NodeInfo))] for n in range(len(NodeInfo))]
	for i in range(0, len(Prob)):
		for j in range(i, len(Prob[i])):
			Prob[i][j] = WeightFunc(NodeInfo[i], NodeInfo[j])
			Prob[j][i] = Prob[i][j]
	return Prob


def HandSeed(Tobimg, img, Surround):
	Upground = set()
	Background = set()
	
	for Kase in range(0, 2):
		Owari = False
		while 1:
			print("\nInput the location you choice as '[i1, j1] [i2, j2]'. \nAfter choose, close the figure and press Enter to continue.")
			if Kase == 0:
				print("Upground Set:  ")
			elif Kase == 1:
				print("Background Set:  ")
			
			#Print the figure
			if Surround == "Nor":
				Pretreatment.FigurePrint(img, 2)
			else:
				pass
			
			#Pretreatment
			InpStr = input()
			RemStr1 = ""
			RemStr2 = ""
			Str2Int = False
			Error = False
			kind = 0

			#Get string
			for i in range(0, len(InpStr)):	
				#Partial
				if InpStr[i] == "[":
					kind = 1
					continue
				if InpStr[i] == ",":
					kind = 2
					continue
				if InpStr[i] == "]":
					kind = 0
					Str2Int = True

				if kind == 1 and Str2Int == False:
					RemStr1 += InpStr[i]
				if kind == 2 and Str2Int == False:
					RemStr2 += InpStr[i]

				if Str2Int == True:
					Int1 = 0
					Int2 = 0
					#print([RemStr1,RemStr2])
					try:
						Int1 = int(RemStr2)
						Int2 = int(RemStr1)
					except:
						print("Input Error, Please input points again")
						Error = True
						break
					
					#print(Tobimg[Int1][Int2])
					try:
						if Kase == 0:
							Upground.add(Tobimg[Int1][Int2])
						if Kase == 1: 
							Background.add(Tobimg[Int1][Int2])

					except:
						print("Input Error, Location exceed")
					
					RemStr1 = ""
					RemStr2 = ""
					Str2Int = False
					continue

			if Error == True:
				continue
			else:
				if Kase == 0:
					if len(Upground) == 0:
						print("You must input at least 1 node!")
						continue
					else:
						Owari = True

				elif Kase == 1:
					if len(Background) == 0:
						print("You must input at least 1 node!")
						continue
					else:
						Owari = True
			if Owari == True:
				break
			else:
				continue

	return Upground, Background




def LinFuncGroup(InpArr, VarN):
	for i in range(0, len(InpArr)):
		Tem = InpArr[i][i]
		for j in range(0, len(InpArr[i])):
			InpArr[i][j] /= Tem

		for p in range(i+1, len(InpArr)):
			Tem = InpArr[p][i]
			for q in range(i, len(InpArr[p])):
				InpArr[p][q] -= Tem * InpArr[i][q]
	
	for i in range(len(InpArr)-1, -1, -1):
		for p in range(i-1, -1, -1):
			Tem = InpArr[p][i]
			for q in range(i, len(InpArr[p])):
				InpArr[p][q] -= Tem * InpArr[i][q]

	ReArr = [[0.00 for n in range(len(InpArr[0]) - VarN)] for n in range(VarN)]
	
	for i in range(0, len(ReArr)):
		for j in range(0, len(ReArr[i])):
			ReArr[i][j] = InpArr[i][j+VarN]
	return ReArr


def MCMCSeed(NodeInfo, LapArr):
	Seedset = set()

	return

def GetProbomatrix(Block,Seedset):
	#By Tao Ren
	len_B = len(Block)
	len_S = len(Seedset)
	#print(seedset)
	SeedBlock =[]
	for each in Seedset:
		SeedBlock.append(Block[each])
	for each in Seedset:
		Block.remove(Block[each])
	Block.remove(Block[0])
	NoSeedBlock = Block
	n = len_B-1
	l = len_S
	wll = np.eye(l)
	wlu = np.zeros((l, n - l))
	wul = np.array([[0] * (l) for each in range(n - l)], dtype=np.float32)
	wuu = np.array([[0] * (n - l) for each in range(n - l)], dtype=np.float32)
	#caculate wul
	i,j = 0,0
	#print(SeedBlock)
	#print(NoSeedBlock)
	#print(len(NoSeedBlock),len(SeedBlock))
	for eachnoseed in NoSeedBlock:
		for eachseed in SeedBlock:
			wul[i,j] = WeightFunc(eachnoseed,eachseed)
			j = j+1
		j = 0
		i = i+1
	# caculate wuu
	i,j = 0,0
	for eachnoseedi in NoSeedBlock:
		for eachnoseedj in NoSeedBlock:
			if i!=j:
				wuu[i,j] = WeightFunc(eachnoseedi,eachnoseedj)
			else:
				wuu[i, j] = 0.01
			j = j+1
		j = 0
		i = i+1

	W = np.vstack((np.hstack((wll, wlu)), np.hstack((wul, wuu))))
	D = np.zeros(W.shape, dtype=np.float32)
	#print('W',W)
	D = np.diag(W.sum(1))
	P = np.linalg.inv(D).dot(W)
	return [[P,n,l],NoSeedBlock]


def GetClassifyProb(Pln):
	#By Tao Ren
	P = Pln[0]
	n = Pln[1]
	l = Pln[2]
	
	# pll = np.eye(l)
	# plu = np.zeros((l, n - l))
	# pul = P[l:n, 0:l]
	# print(pul.shape)
	# puu = P[l:n, l:n]
	# print(puu.shape)
	# # 直接求逆
	# # y = np.linalg.inv((np.eye(puu.shape[0])-puu)).dot(pul)
	# # 迭代m次
	# m = 50
	# Py = np.ones(pul.shape, np.float32)/l
	# for k in range(m):
	#	 Py = puu.dot(Py) + pul
	# print(Py)

	for i in range(20):
		P = P.dot(P)
	Py = P[l:n, 0:l]

	return Py.tolist()


def RWclassify(Py,NoseedBlock):
	#By Tao Ren
	i=0
	for each in Py:
		if max(each)>=0.4:
			each.append(each.index(max(each)))
		else:
			each.append(-1)
		each.append(NoseedBlock[i][0])
		i += 1

	return Py


def BlockFigure(Tobimg, ProbBlock, LenSeed, Upground):
	ProbArr = [0 for n in range(len(ProbBlock) + LenSeed + 10)]
	for i in range(0, len(ProbBlock)):
		ProbArr[ProbBlock[i][len(ProbBlock[i])-1]] = ProbBlock[i][len(ProbBlock[i]) - 2]
	
	UpgSeed = set()
	for i in Upground:
		UpgSeed.add(ProbArr[i])

	for i in range(0, len(ProbBlock)):
		Sign = set([ProbArr[i]])
		if len(Sign & UpgSeed) == 0:
			ProbArr[i] = 0
		else:
			ProbArr[i] = 1

	#print(ProbArr)
	for i in range(0, len(Tobimg)):
		for j in range(0, len(Tobimg[i])):
			if ProbArr[Tobimg[i][j]] == 0:
				Tobimg[i][j] = 0
			else:
				Tobimg[i][j] = 255

	return Tobimg


def GetBoundary(Tobimg):
	"""
	BFS(LocX, LocY, Node)
		This function is used for breath first search algorithm.
		The main idea of this function is traversal all the figure to find the boundary of figure
		LocX = The x Location will judge
		LocY = The y Location will judge
		Node = The Block Code of all the figure, if the node is changed, that means this block is boundary 
		return 1: means the block code had been changed
			   0: means the block has not been changed
	Also, this function will use the functional array ReFig
	"""

	ReFig = [[-1 for n in range(len(Tobimg[0]))] for n in range(len(Tobimg))]
	
	def BFS(LocX, LocY, Node):
		Init.LogWrite("[" + str(LocX) + ", " + str(LocY) + "]", "0")
		if LocX >= len(ReFig) or LocY >= len(ReFig[0]):
			return 0
		
		if ReFig[LocX][LocY] != -1:
			return 0
		if Tobimg[LocX][LocY] != Node:
			return 1

		#print([Tobimg[LocX][LocY], Node])
		if BFS(LocX + 1, LocY, Tobimg[LocX][LocY]) == 1:
			ReFig[LocX][LocY] = 255
		else:
			ReFig[LocX][LocY] = 0
		
		if BFS(LocX, LocY + 1, Tobimg[LocX][LocY]) == 1:
			ReFig[LocX][LocY] = 255
		elif ReFig[LocX][LocY] == -1:
			ReFig[LocX][LocY] = 0
		return 0


	for i in range(0, len(Tobimg)):
		for j in range(0, len(Tobimg[i])):
			if ReFig[i][j] != -1:
				continue
			else:
				if BFS(i + 1, j, Tobimg[i][j]) == 1:
					ReFig[i][j] = 255
				else:
					ReFig[i][j] = 0
				
				if BFS(i, j + 1, Tobimg[i][j]) == 1:
					ReFig[i][j] = 255
				elif ReFig[i][j] == -1:
					ReFig[i][j] = 0
					
	Init.LogWrite("Figure iterator succeed","0")
	return ReFig


def Laplacian(NodeInfo, VarL):
	VarU = len(NodeInfo) - VarL
	Laplacian = [[0.00 for n in range(len(NodeInfo))] for n in range(len(NodeInfo))]
	#print(Laplacian)

	for i in range(0, len(Laplacian)):
		for j in range(i, len(Laplacian)):
			if i == j:
				continue
			else:
				imaWei = WeightFunc(NodeInfo[i], NodeInfo[j])
				Laplacian[i][i] += imaWei
				Laplacian[j][j] += imaWei

				Laplacian[i][j] -= imaWei
				Laplacian[j][i] -= imaWei
	
	#Init.ArrOutput(Laplacian)
	
	#Normalization
	for i in range(0, len(Laplacian)):
		for j in range(i, len(Laplacian[i])):
			if i == j:
				continue
			else:
				Laplacian[i][j] /= Laplacian[i][i]
				Laplacian[j][i] /= Laplacian[j][j]

	for i in range(0, len(Laplacian)):
		Laplacian[i][i] = 1

	ReLap = [[0.00 for n in range(VarL + VarU)] for n in range(VarU)]
	
	for i in range(0, VarU):
		for j in range(0, VarU):
			ReLap[i][j] = Laplacian[i + VarL][j + VarL]

	for i in range(0, VarU):
		for j in range(0, VarL):
			ReLap[i][j + VarU] = Laplacian[i + VarL][j]


	return ReLap


def SobelAlg(img):
	dx = ndimage.sobel(img, 0)
	dy = ndimage.sobel(img, 1)
	img = np.hypot(dx, dy)
	img *= 255.0 / np.max(img)
	return img




