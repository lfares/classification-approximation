import csv
import numpy as np
from sklearn.metrics import mean_squared_error

# Declaration of functions
def logistic(x):
    return .5 * (1 + np.tanh(.5 * x))

def logistic_derivative(x):
    return logistic(x)*(1 - logistic(x))

# Get input and t values from default_features file
list_inputs = []
list_t = []
list_test_inputs = []
list_test_results = []
with open("default_features_1059_tracks.csv", "r") as csv_file:
	csv_reader = csv.reader(csv_file)
	for row in csv_reader:
		list_inputs.append(row[:-2])
		list_t.append(row[-2:])

# Normalize inputs and t values
matrix_inputs = np.array(list_inputs).astype(np.float)
matrix_t = np.array(list_t).astype(np.float)
matrix_inputs = (matrix_inputs - matrix_inputs.min(0)) / matrix_inputs.ptp(0)
matrix_t = (matrix_t - matrix_t.min(0)) / matrix_t.ptp(0)

# Auxiliar variables
epochs_range = [1000, 750, 500] 
learning_rate_range = [0.1, 0.075, 0.05] 
momentum_range = [0.5, 0.25, 0.]
n_tests = [int(1059/3), int(1059/4), int(1059/5)] 

for n_test in n_tests:
	# Get train and test values
	test = np.random.random_integers(0, (1059-1), n_test)
	matrix_test_inputs = matrix_inputs[test, :]
	matrix_test_results = matrix_t[test, :]
	matrix_inputs = np.delete(matrix_inputs, test, 0)
	matrix_t = np.delete(matrix_t, test, 0)

	entries = len(matrix_inputs[0]) # number of features to annalize
	hidden_neurons = entries
	output_number = 2 # latitude and longitude
	for momentum in momentum_range:
		for learning_rate in learning_rate_range:
			for epochs in epochs_range:
				# Initialization of weights, bias and momentum
				weights_hidden = np.random.uniform(low=-0.1, high=0.1, size=(entries, hidden_neurons))
				weights_output = np.random.uniform(low=-0.1, high=0.1, size=(hidden_neurons, output_number))
				# bias_hidden = np.random.uniform(size=(hidden_neurons, 1))
				# bias_output = np.random.uniform(size=(output_number, 1))
				prev_delta_hidden = np.zeros(shape=(entries, hidden_neurons))
				prev_delta_output = np.zeros(shape=(hidden_neurons, output_number))

				### Training ###

				for i in range(epochs):
					# print("Epoch:"+str(i)+"/"+str(epochs))

					# Forward Propagation
					layer_hidden_init = np.dot(matrix_inputs, weights_hidden)
					# layer_hidden_init += bias_hidden
					layer_hidden_output = logistic(layer_hidden_init)

					output_layer_init = np.dot(layer_hidden_output,weights_output)
					# outputLayerInit += bias_output
					predict_output_FP = logistic(output_layer_init)

					# print("Predicted output from forward propagation: ")
					# print(predict_output_FP)

					# Backpropagation
					error_output_layer = matrix_t - predict_output_FP
					delta_output_layer = np.dot(layer_hidden_output.T, error_output_layer*logistic_derivative(predict_output_FP))*learning_rate
					
					error_hidden_layer = error_output_layer.dot(weights_output.T)
					delta_hidden_layer = np.dot(matrix_inputs.T, error_hidden_layer*logistic_derivative(layer_hidden_output))*learning_rate

					# print("Error: ")
					# print(error_output_layer)

					# Updating weights
					weights_output += delta_output_layer + momentum*prev_delta_output
					weights_hidden += delta_hidden_layer + momentum*prev_delta_hidden

					# print("Updated weights output: ")
					# print(weights_output)

					prev_delta_output = delta_output_layer
					prev_delta_hidden = delta_hidden_layer

				### Testing ###

				# Initialize variables
				layer_hidden_init_test = np.dot(matrix_test_inputs, weights_hidden)
				layer_hidden_output_test = logistic(layer_hidden_init_test)
				output_layer_init_test = np.dot(layer_hidden_output_test, weights_output)
				predict_output_test = logistic(output_layer_init_test)

				error = mean_squared_error(matrix_test_results, predict_output_test)

				print("Tested with "+str(n_test)+" values")
				print("Number of cycles (epochs): "+str(epochs))
				print("Learning rate: "+str(learning_rate))
				print("Momentum: "+str(momentum))
				print("Mean squared error: "+str(error))