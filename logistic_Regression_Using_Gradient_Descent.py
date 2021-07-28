"""
trained on ds2_train tested on ds1_test
alpha = 0.7 accu = 74
alpha = 0.8 accu = 78
alpha = 0.5 accu = 64
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def sigmoid(x):
	return (1/(1+np.exp(-1*x)))



data = pd.read_csv('D:/CodeBase/Machine Learning/data/ds1_train.csv')
print(data) 

x = np.array(list(zip(data['x_1'],data['x_2'])))
# print(x)

y = data['y']
# print(y)

weights = np.array([0.001,-0.01])
bias = 0
aplha = 0.01
# print(response)

for epoch in range(10000):
	response = np.array(np.dot(weights,np.transpose(x))+bias)
	for i in range(len(response)):
		t = sigmoid(response[i])
		response[i] = t
	t1 = 0
	t2 = 0
	b = 0
	for i in range(len(y)):
		t1 += (response[i]-y[i])*(x[i][0])
		t2 += (response[i]-y[i])*(x[i][1])
		b += (response[i]-y[i])
	t1 /= len(y)
	t2 /= len(y)
	b /= len(y)
	weights[0] -= aplha*t1
	weights[1] -= aplha*t2
	bias -= aplha*b
	# print("t1:{}, t2:{}".format(t1,t2))

print(weights,bias)


test1 = pd.read_csv('D:/CodeBase/Machine Learning/data/ds1_test.csv')

test1_inputs = np.array(list(zip(test1['x_1'],test1['x_2'])))
# print(test1_inputs)

test1_output = np.array(test1['y'])

test1_pred_output = np.array(np.dot(weights,np.transpose(test1_inputs))+bias)

for i in range(len(test1_pred_output)):
	t = sigmoid(test1_pred_output[i])
	if t >= 0.5 :
		test1_pred_output[i] = 1
	else :
		test1_pred_output[i] = 0
count = 0

for i in range(len(test1_pred_output)):
	if test1_pred_output[i] == test1_output[i]:
		count += 1
	print("The real output : {}, the predicted output : {}".format(test1_output[i],test1_pred_output[i]))
print(count)