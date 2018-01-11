#=================================================================
#
#		CNN and RW Image Edge Algorithm
#		Copyright(c) KazukiAmakawa, all right reserved.
#		Functions.py
#
#=================================================================
"""
		FUNCTION INSTRUCTION
		
Convolution(InpX, InpK)
	This function will convolution two matrix with FFT algorithm

	InpX = The matrix of figure
	InpK = The matrix of kernal

	return Convolved Array

SeedFirst(NodeInfo, SeedSet)
	This function will put the seed node at the first of NodeInfo and return the length of Seedset

	NodeInfo = The node infomation, array, every element have four variable, [Code, Grey, Locx, Locy]
	SeedSet = The set of node which is choosed as seed

	return New NodeInfo which have Seed first, Length of Seeds

LinearEquation(EquArray, VarL)
	This function can solve groups of linear equation goups with elementary transformation.
	for the functions as
	Ax = b_1, Ax = b_2, ..., Ax = b_n, the input s.t.

	EquArray = [A][b_1][b_2]...[b_n]
	VarL = len(x)

	return The solution array
		[[x11, x12, ..., x1n], [x21, x22, ..., x2n], ..., [xn1, xn2, ..., xnn]]

	Note: Actually, in this problem, it is the transformed probability we need to solve.

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

#import files
import Init


def Convolution(InpX, InpK):
	Init.LogWrite("Convolution run succeed","0")
	return signal.fftconvolve(InpX, InpK[::-1], mode='full')


def SeedFirst(NodeInfo, SeedSet):
	RemArr = [False for n in range(len(NodeInfo))]
	NewInfo = []
	VarL = 0
	for SeedLoc in SeedSet:
		VarL += 1
		NewInfo.append(NodeInfo[SeedLoc])
		RemArr[SeedLoc] = True
	
	for i in range(1 , len(RemArr)):
		if RemArr[i] == False:
			NewInfo.append(NodeInfo[i])
	
	return NewInfo, VarL


def LinearEquation(EquArray, NumVar, VarL):
	#Init.ArrOutput(EquArray)
	print()
	for i in range(0, NumVar):
		Rate = str(int(i / NumVar * 10000))
		while len(Rate) != 4:
			Rate = "0" + Rate
		print("Solution Rate: " + Rate[0] + Rate[1] + "." + Rate[2] + Rate[3] + "%", end = "\r")
		Tem = EquArray[i][i]
		for j in range(i, len(EquArray[i])):
			EquArray[i][j] /= Tem

		for p in range(0, len(EquArray)):
			if p == i:
				continue
			else:
				Tem = EquArray[p][i]
				for q in range(i, len(EquArray[p])):
					EquArray[p][q] -= Tem * EquArray[i][q]

	print("Solution Rate: 100.00%")
	Result = [[0.00 for n in range(VarL)] for n in range(NumVar)]
	
	#Init.ArrOutput(Result)
	TTL = [0.00 for n in range(VarL)]
	for i in range(0, len(Result)):
		for j in range(0, len(Result[i])):
			Result[i][j] = abs(EquArray[i][j + NumVar])
			TTL[j] += Result[i][j]
	
	#Init.ArrOutput(Result)
	
	for i in range(0, len(Result)):
		for j in range(0, len(Result[i])):
			Result[i][j] /= TTL[j]
	
	#Init.ArrOutput(Result)
	
	RetArr = []
	for i in range(0, len(Result)):
		Pair = [0, i + VarL]
		maxx = 0
		for j in range(0, len(Result[i])):
			if maxx < Result[i][j]:
				maxx = Result[i][j]
				Pair[0] = j
		RetArr.append(Pair)
	return RetArr










