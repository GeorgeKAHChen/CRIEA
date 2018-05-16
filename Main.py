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
	#Get the seed pixels
	#==============================================================================
	"""
	Seed = Algorithm.GetSeed(img)



	"""
	#==============================================================================
	#Calculate the weight
	#==============================================================================
	"""
	






Main()
