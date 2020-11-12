import numpy as np

input_features = np.array([[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0]])
print("Input array size: {}".format(input_features.shape))
output = np.array([0.0,6.0,5.0,4.0,3.0])
output.reshape(5,1)
print("Output array: {}".format(output))
print("Input array: {}".format(input_features))
weights = np.array([0.5,1.5,2.5])
weights = weights.reshape(1,3)
print("Weights {} and weights size {}".format(weights,weights.shape))
lr = 0.05
for epoch in range (1000):
	pred_output = np.dot(weights,input_features.T)
	pred_output.reshape(5,1)
	print("Predicted outputs: {}".format(pred_output))
	error = np.array(pred_output-output)
	error.reshape(1,5)
	d = np.dot(error,input_features)
	for i in range(3):
		weights[0][i] -= float(d[0][i]*lr)/float(5)
	print("Weights after updates: {}".format(weights))

