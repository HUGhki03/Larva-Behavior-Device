from region_shihan import Region
import numpy as np

# creates the state machine and defines the seven states: the center of the maze, the 3 channels, and the 3 circles

class StateMachine:
	def __init__(self,initial, locs):
		self.initial = initial
		self.locs = locs
		self.state = Initializer(self.initial,self.locs).starting_point(self.initial)
	def on_input(self, input):
		self.prevnumber = self.state.mynumber
		if input == [-1,-1]: #if tracker lost larva, keep previous state
			try:
					self.prevnumber
			except NameError:
					return 0, 0, 0
			else:
					return self.prevnumber, self.prevnumber, self.prevnumber
		else:
			self.state = self.state.on_input(input)
			self.number = self.state.getState(input)
			self.nextnum = self.state.getnextclosestState(input)
			return self.prevnumber, self.number, self.nextnum

class Initializer(Region):
	def starting_point(self, input):
		states = {0 : Zero(self.locs[0], self.locs), 1 : One(self.locs[1],self.locs), 2 : Two(self.locs[2],self.locs), 3: Three(self.locs[3],self.locs), 4: Four(self.locs[4],self.locs), 5: Five(self.locs[5],self.locs), 6: Six(self.locs[6], self.locs)}
		state = self.getState(input)
		return states[state]

class Zero(Region):
	mynumber = 0
	def on_input(self, input):
		if input != [-1,-1]:
			states = {0 : self, 1 : One(self.locs[1],self.locs), 2 : Two(self.locs[2],self.locs), 3: Three(self.locs[3],self.locs), 4: Four(self.locs[4],self.locs), 5: Five(self.locs[5],self.locs), 6: Six(self.locs[6], self.locs)}
			transition = self.processPoint(input)
			state = self.getState(input)
			if transition == True:
				return states[state]
			else:
				return self

class One(Region):
	mynumber = 1
	def on_input(self, input):
		if input != [-1,-1]:
			states = {0 : Zero(self.locs[0],self.locs), 1 : self, 2 : Two(self.locs[2],self.locs), 3: Three(self.locs[3],self.locs), 4: Four(self.locs[4],self.locs), 5: Five(self.locs[5],self.locs), 6: Six(self.locs[6], self.locs)}
			transition = self.processPoint(input)
			state = self.getState(input)
			if transition == True:
				return states[state]
			else:
				return self

class Two(Region):
	mynumber = 2
	def on_input(self, input):
		if input != [-1,-1]:
			states = {0 : Zero(self.locs[0], self.locs), 1 : One(self.locs[1],self.locs), 2 : self, 3: Three(self.locs[3],self.locs), 4: Four(self.locs[4],self.locs), 5: Five(self.locs[5],self.locs), 6: Six(self.locs[6], self.locs)}
			transition = self.processPoint(input)
			state = self.getState(input)
			if transition == True:
				return states[state]
			else:
				return self

class Three(Region):
	mynumber = 3
	def on_input(self, input):
		if input != [-1,-1]:
			states = {0 : Zero(self.locs[0], self.locs), 1 : One(self.locs[1],self.locs), 2 : Two(self.locs[2],self.locs), 3: self, 4: Four(self.locs[4],self.locs), 5: Five(self.locs[5],self.locs), 6: Six(self.locs[6], self.locs)}
			transition = self.processPoint(input)
			state = self.getState(input)
			if transition == True:
				return states[state]
			else:
				return self
	
class Four(Region):
	mynumber = 4
	def on_input(self, input):
		if input != [-1,-1]:
			states = {0 : Zero(self.locs[0], self.locs), 1 : One(self.locs[1],self.locs), 2 : Two(self.locs[2],self.locs), 3: Three(self.locs[3],self.locs), 4: self, 5: Five(self.locs[5],self.locs), 6: Six(self.locs[6], self.locs)}
			transition = self.processPoint(input)
			state = self.getState(input)
			if transition == True:
				return states[state]
			else:
				return self

class Five(Region):
	mynumber = 5
	def on_input(self, input):
		if input != [-1,-1]:
			states = {0 : Zero(self.locs[0], self.locs), 1 : One(self.locs[1],self.locs), 2 : Two(self.locs[2],self.locs), 3: Three(self.locs[3],self.locs), 4: Four(self.locs[4],self.locs), 5: self, 6: Six(self.locs[6], self.locs)}
			transition = self.processPoint(input)
			state = self.getState(input)
			if transition == True:
				return states[state]
			else:
				return self

class Six(Region):
	mynumber = 6
	def on_input(self, input):
		if input !=[-1,-1]:
			states = {0 : Zero(self.locs[0], self.locs), 1 : One(self.locs[1],self.locs), 2 : Two(self.locs[2],self.locs), 3: Three(self.locs[3],self.locs), 4: Four(self.locs[4],self.locs), 5: Five(self.locs[5],self.locs), 6: self}
			transition = self.processPoint(input)
			state = self.getState(input)
			if transition == True:
				return states[state]
			else:
				return self

