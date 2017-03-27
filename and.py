import numpy as np

#Sigmoid function
#used as activation function
def sigmoid(x):
	return (1/(1+np.exp(-x)))

#initialize random weights
def init_weight(input_no,percetron_no, output_no):
	i_h_weights = np.random.normal(0,1/(number_of_inputs**2),size=(number_of_inputs,number_of_hidden_perceptron))
	h_o_weights = np.random.normal(0,1/(number_of_inputs**2), size=(number_of_hidden_perceptron, number_of_outputs))
	return i_h_weights,h_o_weights

#inputs
X = np.array([
	[0,0],
	[0,1],
	[1,0],
	[1,1]
	])

#outputs
Y = np.array([
	[0],
	[0],
	[0],
	[1]
	])

#params
number_of_inputs = 2
number_of_hidden_perceptron = 4
number_of_outputs = 1
epochs = 600000
learning_rate = 0.3

#For consistent testing
np.random.seed(1)

input_to_hidden_weights, hidden_to_output_weights = init_weight(number_of_inputs, number_of_hidden_perceptron, number_of_outputs)

# print (input_to_hidden_weights)
for e in range(epochs):
	#inputs for the hidden layer
	hidden_layer_in = np.dot(X, input_to_hidden_weights)

	#activation function output of hidden layer
	hidden_layer_out = sigmoid(hidden_layer_in)

	#inputs for output layer
	output_layer_in = np.dot(hidden_layer_out,hidden_to_output_weights)

	#activation function output of output layers
	output_layer_out = sigmoid(output_layer_in)

	#Error calculation
	error = Y - output_layer_out

	#Backward propagation
	del_err_output = error * output_layer_out * (1 - output_layer_out)


	del_err_hidden = np.dot(del_err_output, hidden_to_output_weights.T) * hidden_layer_out * (1 - hidden_layer_out)

	#update parameter for weights (delta weights)
	delta_input_hidden = learning_rate * np.dot(X.T, del_err_hidden)
	delta_hidden_output = learning_rate * np.dot(hidden_layer_out.T,del_err_output)
	
	#update weights
	input_to_hidden_weights += delta_input_hidden
	hidden_to_output_weights += delta_hidden_output

	if(e%(epochs/10) == 0):	
		print ("After ",e, " iterations\n",output_layer_out)

# print (input_to_hidden_weights, hidden_to_output_weights)