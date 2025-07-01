import numpy as np
import cv2

# creates the Region class; contains functions for processing the location of the larva and determining the larva’s state from it’s location

class Region(object):
	def __init__(self, loc, locs):
		self.loc = loc #location of this region
		self.locs = locs #location of all regions
		self.arm_width = 35
		self.circle_radius = 66

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

	def getRectangleRegions(self):

		# 根据maze中心点和圆心点计算臂范围的矩形四个顶点
		arm_width = self.arm_width
		rec_points = []
		for i in range(4,7):
			dx = self.locs[i][0] - self.locs[0][0]
			dy = self.locs[i][1] - self.locs[0][1]
			
			# 垂直向量，模
			dy2 = -dx/(dx**2 + dy**2)**0.5
			dx2 = dy/(dx**2 + dy**2)**0.5
			vec_mod = (dx2**2 + dy2**2)**0.5

			# 矩形四个点的坐标
			point1_x = int(round(self.locs[0][0] + arm_width * 0.5 * dx2/ vec_mod))
			point1_y = int(round(self.locs[0][1] + arm_width * 0.5 * dy2/ vec_mod))
			point2_x = int(round(self.locs[0][0] - arm_width * 0.5 * dx2/ vec_mod))
			point2_y = int(round(self.locs[0][1] - arm_width * 0.5 * dy2/ vec_mod))
			point3_x = int(round(self.locs[i][0] - arm_width * 0.5 * dx2/ vec_mod))
			point3_y = int(round(self.locs[i][1] - arm_width * 0.5 * dy2/ vec_mod))
			point4_x = int(round(self.locs[i][0] + arm_width * 0.5 * dx2/ vec_mod))
			point4_y = int(round(self.locs[i][1] + arm_width * 0.5 * dy2/ vec_mod))               
			
			rec_points.append((np.array([[point1_x,point1_y],[point2_x,point2_y],[point3_x,point3_y],
							[point4_x,point4_y]], dtype = np.int32)).reshape(4,1,2))
			
		return rec_points
	
	def isinRegions(self,x):
		# 判断是否在矩形范围内
		rec_points = self.getRectangleRegions()
		for points in rec_points:
			if cv2.pointPolygonTest(points, x, False) > 0:
				return True
		# 判断是否在圆形范围内
		cir_centers = self.locs[4:]
		ndist = self.distance(x,cir_centers)
		return np.amin(ndist) < self.circle_radius
	
	def drawRegions(self,image):
		# 在图像上绘制区域范围
		rec_points = self.getRectangleRegions()
		edge_color = (128,128,128)
		for points in rec_points:
			cv2.polylines(image, [points], isClosed=True, color=edge_color, thickness=2)
        
		for cir_center in self.locs[4:]:
			cv2.circle(image, cir_center, self.circle_radius, edge_color, 2)
		
		return image