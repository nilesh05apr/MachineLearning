import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def frand():
	return np.random.normal(0.0,1.0)

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-1*x))


class Perceptron:
	def __init__(self,inputs,bias = 1.0):
		self.bias = bias
		self.weights = np.zeros(inputs+1)
		for i in range(inputs):
			self.weights[i] = frand()
	def run(self,X):
		X.append(self.bias)
		s = np.dot(np.transpose(self.weights),X)
		return sigmoid(s)
	def set_weight(self,w_init):
		self.weights = w_init


print("\n\n And Gate:\n")

nn = Perceptron(2)
nn.set_weight(np.array([10.0,10.0,-15]))
print("0 And 0: {}".format(nn.run([0.0,0.0])))
print("0 And 1: {}".format(nn.run([0.0,1.0])))
print("1 And 0: {}".format(nn.run([1.0,0.0])))
print("1 And 1: {}".format(nn.run([1.0,1.0])))