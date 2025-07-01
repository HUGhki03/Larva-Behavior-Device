import numpy as np
import cv2
from collections import deque

# creating and updating the background

class BakCreator:
	def __init__(self,stacklen,alpha,bgim):
		self.stacklen = stacklen
		self.alpha = alpha     #alpha = percent change
		self.bgim = bgim      #initialize bgim with initial background image
		self.Ims = FIFO(self.stacklen, 'Ims')
		self.Tims = FIFO(10, 'Tims')
		self.updatetime = 0
	def orStack(self,stack):  #对栈中所有图像按位或运算（乘积）
		if stack:
			if len(stack)==1:
				print("Premature Stack")
				return stack[0]
			else:
				a = stack[0]
				for i in range(len(stack)-1):
					or_i = cv2.bitwise_or(a,stack[i+1])  # Calculates the per-element bit-wise disjunction of two arrays or an array and a scalar.
					a = or_i	
				return or_i
		else:
			print("Problem wih orStack: input stack is empty")
	def makeORStack(self):
		self.timsOR = self.orStack(self.Tims)
		print("Initial OR stack created")
	def update(self,newIm,fgthresh):
		self.Ims.add(newIm)
		self.Tims.add(fgthresh)
		if self.Ims.loading == False:
			self.bgim = np.median(self.Ims.stack,axis=0).astype(dtype = np.uint8)
	def run(self,newIm,fgthresh):
		if self.Tims.loading:
			self.Tims.add(fgthresh)
		else:
			self.timsOR = self.orStack(self.Tims.stack)
			numNewPix = cv2.countNonZero(cv2.bitwise_and(fgthresh,cv2.bitwise_not(self.timsOR)))
			if numNewPix > (self.alpha*cv2.countNonZero(fgthresh)):
				self.update(newIm,fgthresh)
				self.updatetime = 0
			else:
				self.updatetime += 1
		return self.bgim


class FIFO:
	def __init__(self,maxlength,name):
		self.maxlength = maxlength
		self.name = name
		self.stack = deque()
		self.loading = True
	def getLength(self):
		return len(self.stack)
	def add(self,im):
		length = self.getLength()
		if length < self.maxlength:
			self.stack.append(im)
		else:
			self.loading = False
			self.stack.popleft()
			self.stack.append(im)
	

