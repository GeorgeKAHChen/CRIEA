#=================================================================
#
#		CNN and RW Image Edge Algorithm
#		Copyright(c) KazukiAmakawa, all right reserved.
#		Works.py
#
#=================================================================
"""
This File have no relationship of the main part. It is just using for 
some test.
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
import random

#import files
import Init
import Pretreatment
import Algorithm
import Functions


def MainFunction():
	if os.path.exists("Output/"):
		if Init.SystemJudge() == 0:
			os.system("rm -r Output")
		else:
			os.system("rmdir /s /q directory")

	NameArr = Pretreatment.FigureInput(1)
	try:
		if NameArr == -1:
			return
	except:
		pass

	#Figure traversal
	for kase in range(0, len(NameArr)):
		img = np.array(Image.open(NameArr[kase]).convert("L"))	
		img1 = img.deepcopy()

		#Algorithm1



def BWError():
	if os.path.exists("Output/"):
		if Init.SystemJudge() == 0:
			os.system("rm -r Output")
		else:
			os.system("rmdir /s /q directory")

	NameArr = Pretreatment.FigureInput(1)
	try:
		if NameArr == -1:
			return
	except:
		pass

	#Figure traversal
	for kase in range(0, len(NameArr)):
		img = np.array(Image.open(NameArr[kase]).convert("L"))	
		"""		
		for i in range(0, len(img)):
			for j in range(0, len(img[i])):
				if random.randint(1, 50) == 1:
					img[i][j] += np.random.normal(img[i][j], 64)
					img[i][j] = max(0, img[i][j])
					img[i][j] = min(255, img[i][j])
		"""
		Name = "Figure_"
		Name += str(Init.GetTime())
		Name += ".png"
		Pretreatment.Output(img, Name, 2)


def FouriorTrans():
	if os.path.exists("Output/"):
		if Init.SystemJudge() == 0:
			os.system("rm -r Output")
		else:
			os.system("rmdir /s /q directory")

	NameArr = Pretreatment.FigureInput(1)
	try:
		if NameArr == -1:
			return
	except:
		pass

	#Figure traversal
	for kase in range(0, len(NameArr)):
		img = np.array(Image.open(NameArr[kase]).convert("L"))	
		Statistic = [0 for n in range(0, 260)]
		TTL = 0
		for i in range(0, len(img)):
			for j in range(0, len(img[i])):
				Statistic[img[i][j]] += 1
				TTL += 1

		#Drecrete PDE
		Prob = [0.00 for n in range(260)]
		for i in range(0, len(Prob)):
			Prob[i] = Statistic[i] / TTL
		
		#HF = np.fft.fft(Prob).real
		
		fig1 = plt.figure()
		ax = fig1.add_subplot(111)
		
		plt.xlim(-1, 260)
		plt.ylim(0, 0.2)

		#Printing loop
		for i in range(0, len(Prob)):
			ax.add_patch(patches.Rectangle((i, 0), 1, Prob[i], color = 'black'))

		Name = ""
		Hajimari = False
		for i in range(len(NameArr[kase]) - 1, -1, -1):
			if NameArr[kase][i] == ".":
				Hajimari = True
				continue
			elif NameArr[kase][i] == "/":
				break
			else:
				if Hajimari == False:
					continue
				else:
					Name = NameArr[kase][i] + Name

		Name += "_Histogram.png"
		print(Name)
		plt.savefig(Name)
		
	return

FouriorTrans()
