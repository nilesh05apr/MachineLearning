import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv('D:/CodeBase/Machine Learning/data/ds2_train.csv')
X = np.array(list(zip(data['x_1'],data['x_2'])))
Y = np.array(data['y'])

m = len(Y)

phi = 0
for i in range(m):
	if Y[i] == 1:
		phi += 1
phi /= m
print("Phi: {}".format(phi))

Mu0 = np.zeros((2,))
Mu1 = np.zeros((2,))


def Mu(l,x):
	t = 0
	for i in range(m):
		if Y[i] == x:
			l[0] += X[i][0]
			l[1] += X[i][1]
	for i in range(m):
		if Y[i] == x:
			t += 1
	l[0] /= t
	l[1] /= t

Mu(Mu0,0)
Mu(Mu1,1)

print("Mu for 0: {}".format(Mu0))
print("Mu for 1: {}".format(Mu1))
sigma = np.zeros((2,2))
Mu0.reshape(2,1)
Mu1.reshape(2,1)

for i in range(m):
	if Y[i] == 0:
		sigma += np.dot((np.transpose(X[i].reshape(1,2))-Mu0),np.transpose(np.transpose(X[i].reshape(1,2))-Mu0))
	else:
		sigma += np.dot((np.transpose(X[i].reshape(1,2))-Mu1),np.transpose(np.transpose(X[i].reshape(1,2))-Mu1))
sigma /= m

print("Sigma : {}".format(sigma))
theta = np.linalg.inv(sigma).dot(Mu1-Mu0)
print("Theta: {}".format(theta))
theta0 = -1*np.log((1-phi)/phi)+0.5*((Mu0.T.dot(np.linalg.inv(sigma))).dot(Mu0)-Mu1.T.dot(np.linalg.inv(sigma)).dot(Mu1))
print("Theta 0: {}".format(theta0))
def sigmoid(x):
	return (1/(1+(np.exp(-1*x))))




test1 = pd.read_csv('D:/CodeBase/Machine Learning/data/ds2_test.csv')
test1_inputs = np.array(list(zip(test1['x_1'],test1['x_2'])))
test1_output = np.array(test1['y'])


pred = []
pred_ = []
for i in range(len(test1_output)):
	t = test1_inputs[i].dot(theta)+theta0
	tx = sigmoid(t)
	pred_.append(tx)
	if tx >= 0.5:
		pred.append(1)
	else:
		pred.append(0)
acc = 0
for i in range(len(test1_output)):
	if pred[i] == test1_output[i]:
		acc += 1
print("Accuracy: {}".format(acc))

