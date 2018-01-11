#=================================================================
#
#		CNN and RW Image Edge Algorithm
#		Copyright(c) KazukiAmakawa, all right reserved.
#		Tools.py
#
#=================================================================
"""
		FUNCTION INSTRUCTION
		
This file will give some tools to make the code suitable on vps.

CNNSeed(TogImg)
	This function will print figure with sign, after you input
	the location of figure, this function will return the .json file
	to tell you figure and situation


"""

import Init
import Pretreatment

def CNNSeed(img, TobImg, BlockSize, FileName):
	BlockInfo = [0 for n in range(BlockSize)]
	for i in range(0, len(BlockSize)):
		print(str(BlockSize) + "\t:")
		OutImg = [[0 for n in ragne(len(TobImg[0]))] for n in range(TobImg)]
		for p in range(0, len(TobImg)):
			for q in range(0, len(TobImg[p])):
				if TobImg[p][q] == i:
					OutImg[p][q] = img[p][q]
		Pretreatment.FigurePrint(OutImg, kind)
		InpInt = Init.IntInput(str(BlockSize) + "\t:" , "1", "3", "int")
		if InpInt == 1:
			BlockInfo[i] = 1
		elif InpInt == 2:
			BlockInfo[i] = 2
		elif InpInt == 3:
			BlockInfo[i] = 3

	
