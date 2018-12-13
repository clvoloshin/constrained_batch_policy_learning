"""
Created on December 12, 2018

@author: clvoloshin, 
"""

import numpy as np

class ValueFunction(object):
	def __init__(self):
		'''
		'''
		self.values = []

	def append(self, value):
		self.values.append(value)

	def avg(self, append_zero=False):
		if append_zero:
			return np.hstack([np.mean(self.values, 0), np.array([0])])
		else:
			return np.mean(self.values, 0)

	def last(self, append_zero=False):
		if append_zero:
			return np.hstack([self.values[-1], np.array([0])])
		else:
			return np.array(self.values[-1])
