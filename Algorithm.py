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
