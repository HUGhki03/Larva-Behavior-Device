import numpy as np

# creates the Region class; contains functions for processing the location of the larva and determining the larva’s state from it’s location

class Region(object):
	def __init__(self, loc, locs):
		self.loc = loc #location of this region
		self.locs = locs #location of all regions
	def getNeighbors(self):
		i = self.locs.index(self.loc)
		del self.locs[i]
		return self.locs
	def distance(self, a, b): #a is the location of larva; b is other point or vector of points
		a = np.asarray(a)
		b = np.asarray(b)
		if a.shape == b.shape:
			displacement = a-b
			return np.sqrt(np.dot(displacement, displacement))
		else:
			A = np.ones_like(b)*a
			displacement = A-b
			c = []
			for i in displacement: 
				distance = (np.sqrt(np.dot(i,i)))
				c.append(distance)
			return np.array(c)
	def processPoint(self, x):
		hyst = 20
		neighbors = self.locs 
		mydist = self.distance(x, self.loc)
		ndist = self.distance(x, neighbors)
		closest = np.amin(ndist)
		if abs(mydist-hyst) > closest and closest != mydist:
			transition = True
		else:
			transition = False
		return transition
	def getState(self,x): #gets current state of the larva
		neighbors = self.locs 
		ndist = self.distance(x, neighbors)
		closest = np.amin(ndist)
		ndist_list = np.ndarray.tolist(ndist)
		state = ndist_list.index(closest)
		return state
	def getnextclosestState(self,x): #gets state larva is closest to
		neighbors = self.locs
		ndist = self.distance(x,neighbors)
		closest = np.amin(ndist)
		m, sm = float('inf'), float('inf')
		for x in ndist:
			if x <= m:
				m, sm = x, m
			elif x < sm:
				sm = x
		ndist_list = np.ndarray.tolist(ndist)
		state = ndist_list.index(sm)


