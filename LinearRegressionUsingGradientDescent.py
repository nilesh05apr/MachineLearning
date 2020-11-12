import numpy as np

input_features = np.array([1.0,2.0,3.0,4.0,5.0,6.0])

output = np.array([-1.0,-2.0,-3.0,-4.0,-5.0,-6.0])

weight = 0.415

lr = 0.05

for i in range(10000):
	pred_outpout = np.dot(np.array(weight),input_features)
	print("Predicted output {}".format(pred_outpout))
	meta_error = 0;
	
  for j in range(len(output)):
		meta_error+= ((pred_outpout[j]-output[j])*input_features[j])/len(output)
	
  weight -= lr*(meta_error)
	print("Weights ({})".format(weight))

test_input = np.array([7.0,8.0,9.0])

test_output = np.dot(np.array(weight),test_input)

print("Test results {}".format(test_output))
