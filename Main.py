#=================================================================
#
#		CNN and RW Image Edge Algorithm
#		Copyright(c) KazukiAmakawa, all right reserved.
#		Main.py
#
#=================================================================
"""
		FUNCTION INSTRUCTION
		
This is the main function of all the program.

MainFunction()
	This is the beginning of all program

	return None
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
import Pretreatment
import Algorithm
import Functions


def MainFunction():
	#CRIEA Start!
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
		img = Pretreatment.BFSmooth(img)
		
		[Tobimg, NodeInfo] = Algorithm.Toboggan(img)
		[Upground, Background] = Algorithm.HandSeed(Tobimg, img, Surround)
		Seeds = Upground | Background		
		ProbBlock = []
		VarL = 0
		
		if Method == "Lap":
			NodeInfo, VarL = Functions.SeedFirst(NodeInfo, Seeds)
			LapEqu = Algorithm.Laplacian(NodeInfo, VarL)
			ProbBlock = Functions.LinearEquation(LapEqu, len(NodeInfo) - VarL, VarL)
		"""
		BlockInfo = Pretreatment.Partial(img)
		BlockSize = len(BlockInfo) - 1

		#MainLoop
		for i in range(1, BlockSize+1):
			if Init.SystemJudge() == 0:
				os.system("clear")
			else:
				os.system("cls")

			print("Area:\t" + str(i) + "/\t" + str(BlockSize))
			#Pre-treatment
			img = np.array(Image.open("Output/Block_" + str(i) + ".png").convert("L"))
			[Tobimg, NodeInfo] = Algorithm.Toboggan(img)
			#Init.ArrOutput(Tobimg)

			[Upground, Background] = Algorithm.HandSeed(Tobimg, img, Surround)
			Seeds = Upground | Background
			
			ProbBlock = []
			VarL = 0

			#Algorithm
			if Method == "Pob":
				Probmat_Nseed = Algorithm.GetProbomatrix(NodeInfo, Seeds)
				ClassifyProb = Algorithm.GetClassifyProb(Probmat_Nseed[0])
				ProbBlock = Algorithm.RWclassify(ClassifyProb,Probmat_Nseed[1])
			
			if Method == "Lap":
				NodeInfo, VarL = Functions.SeedFirst(NodeInfo, Seeds)
				LapEqu = Algorithm.Laplacian(NodeInfo, VarL)
				ProbBlock = Functions.LinearEquation(LapEqu, len(NodeInfo) - VarL, VarL)
			
			#After-treatment
			Figure = Algorithm.BlockFigure(Tobimg, ProbBlock, VarL, Upground)
			#Init.ArrOutput(Figure)
			Figure = cv2.Canny(np.uint8(Figure), 85, 170)
			#Figure = Algorithm.GetBoundary(Figure)

			#Output
			if Surround == "Nor":
				Pretreatment.FigurePrint(Figure, 2)
			else:
				pass
			Pretreatment.Output(Figure, "Block_" + str(i) + ".png", 1)

		Pretreatment.Recovery(BlockSize, BlockInfo)
	"""

MainFunction()


"""
#Camera Man
Upground
[50,122][132,143][53,190][154,131]
Background
[169,43][175,184][4,153]

#Easy Figure
Upground
[25,25]
Background
[50,50]

#Heart-Low
Upground Set:
[37,18][39,64][39,30][81,57][90,78][65,65][43,21][14,52]
Background Set:
[112,25][43,108][100,40][129,99][17,80]

Upground Set:
[42,12][17,56][51,37][82,48][68,73][38,41]
Background Set:
[112,25][43,108][115,45]

Upground Set:
[81,46][66,63][89,66][84,76][79,63]
Background Set:
[81,32][57,60][79,97][109,68]

Upground Set:
[37,38][81,61][35,62]
Background Set:
[126,29][31,108][12,92][125,51][142,61][64,6][98,8][134,17]

Factory_Low
Upground Set:
[96,76]
Background Set:
[120,92]

Upground Set:
[96,76]
Background Set:
[123,34]

Alpha = 0.005
Beta = 0.007

Upground Set:
[98,72][32,174]
Background Set:
[77,130][120,35]
"""