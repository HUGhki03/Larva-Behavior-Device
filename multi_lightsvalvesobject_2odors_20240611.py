import numpy as np
import random
from collections import Counter



# training schemes and for changing airflow based on the larva’s location

class LightsValves:
	def __init__(self,task,v1,v2,v3,v4,v5,v6,lock,name):
		self.name = name
		self.task = task
		self.v1 = v1
		self.v2 = v2
		self.v3 = v3
		self.v4 = v4
		self.v5 = v5
		self.v6 = v6
		self.lock = lock
		self.framecount = 0
		self.larva_loc = 0
		self.elapsedtime = 0

		
	def on(self, pin):
		with self.lock:
			states = self.task.read()
			states[pin] = True
			self.task.write(states)
	def off(self, pin):
		with self.lock:
			states = self.task.read()
			states[pin] = False
			self.task.write(states)
	def get_state(self,pin):
		with self.lock:
			states = self.task.read()
			return states[pin]

	def run_test_noreward(self,prev_state, curr_state, nextclosest_state, decisionlist, decisionlog,elapsedtime,framecount,maggot,odorontime, valvelog):
		self.prev_state = prev_state
		self.curr_state = curr_state
		self.nextclosest_state = nextclosest_state
		self.decisionlist = decisionlist
		self.decisionlog = decisionlog
		self.elapsedtime = elapsedtime
		self.framecount = framecount
		self.maggot = maggot
		self.odorontime = odorontime
		self.valvelog = valvelog
		odor1 = [self.v1, self.v3, self.v5]
		odor2 = [self.v2, self.v4, self.v6]
		odor1state = []
		odor2state = []
		videoframerate = 30

		for i in range(0,3):
			states = self.task.read()
			b = states[odor1[i]]
			odor1state.append(b)
		for i in range(0,3):
			states = self.task.read()
			b = states[odor2[i]]
			odor2state.append(b)

		timevideo = framecount/videoframerate
		timevideomin = int(round(timevideo // 60))
		timevideosec = int(round(timevideo % 60))
		videotimestring = f"{timevideomin:02d}:{timevideosec:02d}" #记录在视频中的对应时间，以mm:ss格式

		if sum(odor1state) + sum(odor2state) == 0: #initial, no odor
			if curr_state in [4,5,6]:
				self.larva_loc = curr_state - 3 #记录初始位置
				odor1_choice = random.randint(1,2) #除现在幼虫所在通道外，其余2通道各开一个气味
				odor1_on = (curr_state - 4 + odor1_choice) % 3
				odor2_on = (curr_state - 1 - odor1_choice) % 3
				self.on(odor1[odor1_on])
				self.on(odor2[odor2_on])
				print(self.name + " Odor1 On Channel "+ str(odor1_on + 1))
				print(self.name + " Odor2 On Channel "+ str(odor2_on + 1))
				
				self.valvelog.append(["Odor1 On - Channel "+ str(odor1_on + 1),str(framecount),str(elapsedtime),videotimestring])
				self.valvelog.append(["Odor2 On - Channel "+ str(odor2_on + 1),str(framecount),str(elapsedtime),videotimestring])

		if sum(odor1state)==1 and sum(odor2state)==1 and curr_state % 3 != self.larva_loc: #正常情况，两种气味都开一个
			odor1_on = odor1state.index(1)
			odor2_on = odor2state.index(1)


			if curr_state-4 == odor1_on: #选择并进入odor1园区并且还没有记录选择
				
				print(self.name + " Choose Odor1 " + str(framecount)+ ", time = " + str(elapsedtime))
				self.decisionlist.append(1)
				self.decisionlog.append((["Choose Odor1 ",str(framecount),str(elapsedtime),videotimestring]))
				self.offall()


			if curr_state-4 == odor2_on: #选择并进入odor2园区并且还没有记录选择
				
				print(self.name + " Choose Odor2 " + str(framecount)+ ", time = " + str(elapsedtime))
				self.decisionlist.append(2)
				self.decisionlog.append((["Choose Odor2 ",str(framecount),str(elapsedtime),videotimestring]))
				self.offall()
		
		

		if sum(odor1state)>1 or sum(odor2state)>1: #错误情况，一种气味开了两个阀
			self.offall()
			print("错误情况:一种气味开了两个阀!!!!!但是已经都关掉啦：）")
			self.decisionlist.append(6)
			self.decisionlog.append((["U R so 6 ",str(framecount),str(elapsedtime),videotimestring]))

		

	def offall(self):
		CO2 = [self.v1, self.v3, self.v5]
		vac = [self.v2, self.v4, self.v6]
		for i in range(0,3):
			self.off(CO2[i])
			self.off(vac[i])
		



