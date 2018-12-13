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
		self.value.append(value)

	def avg(self):
		return np.mean(self.value, 0)

	def last(self):
		return self.value[-1]
