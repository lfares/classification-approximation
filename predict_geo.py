import csv
import numpy as np

# Declaration of functions
def logistic(x):
    return .5 * (1 + np.tanh(.5 * x))

def logistic_derivative(x):
    return logistic(x)*(1 - logistic(x))

# Define number of values to train (the rest of them will be to test)
train = 1050

# Get input and t values from default_features file
list_inputs = []
list_t = []
list_test_inputs = []
list_test_results = []
with open("default_features_1059_tracks.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count < train:
            list_inputs.append(row[:-2])
            list_t.append(row[-2:])
        else:
            list_test_inputs.append(row[:-2])
            list_test_results.append(row[-2:])
        line_count += 1

# print(list_inputs)
print(list_t)

# Change input and t value from lists to numpy matrix to train and test later
matrix_inputs = np.array(list_inputs).astype(np.float)
matrix_t = np.array(list_t).astype(np.float)
matrix_test_inputs = np.array(list_test_inputs).astype(np.float)
matrix_test_results = np.array(list_test_results).astype(np.float)

# Normalizing 
matrix_inputs = (matrix_inputs - matrix_inputs.min(0)) / matrix_inputs.ptp(0)
matrix_test_inputs = (matrix_test_inputs - matrix_test_inputs.min(0)) / matrix_test_inputs.ptp(0)
matrix_t = (matrix_t - matrix_t.min(0)) / matrix_t.ptp(0)
matrix_test_results = (matrix_test_results - matrix_test_results.min(0)) / matrix_test_results.ptp(0)

# Auxiliar variables
epochs = 20000 # 500, 750, 1000 to test the NN
learning_rate = 0.1 #0.05, 0.075 and 0.1 to test the NN
momentum = 0.5 # 0., 0.25 and 0.5 to test the NN 
entries = len(matrix_inputs[0]) # number of features to annalize
hidden_neurons = entries
output_number = 2 # latitude and longitude

# Initialization of weights, bias and momentum
weights_hidden = np.random.uniform(low=-0.1, high=0.1, size=(entries, hidden_neurons))
weights_output = np.random.uniform(low=-0.1, high=0.1, size=(hidden_neurons, output_number))
# bias_hidden = np.random.uniform(size=(hidden_neurons, 1))
# bias_output = np.random.uniform(size=(output_number, 1))
prev_delta_hidden = np.zeros(shape=(entries, hidden_neurons))
prev_delta_output = np.zeros(shape=(hidden_neurons, output_number))

### Training ###

for i in range(epochs):
	print("Epoch:"+str(i)+"/"+str(epochs))

	# Forward Propagation
	layer_hidden_init = np.dot(matrix_inputs, weights_hidden)
	# layer_hidden_init += bias_hidden
	layer_hidden_output = logistic(layer_hidden_init)

	output_layer_init = np.dot(layer_hidden_output,weights_output)
	# outputLayerInit += bias_output
	predict_output_FP = logistic(output_layer_init)

	print("Predicted output from forward propagation: ")
	print(predict_output_FP)

	# Backpropagation
	error_output_layer = matrix_t - predict_output_FP
	delta_output_layer = np.dot(layer_hidden_output.T, error_output_layer*logistic_derivative(predict_output_FP))*learning_rate
	
	error_hidden_layer = error_output_layer.dot(weights_output.T)
	delta_hidden_layer = np.dot(matrix_inputs.T, error_hidden_layer*logistic_derivative(layer_hidden_output))*learning_rate

	print("Error: ")
	print(error_output_layer)

	# Updating weights
	weights_output += delta_output_layer + momentum*prev_delta_output
	weights_hidden += delta_hidden_layer + momentum*prev_delta_hidden

	# print("Updated weights output: ")
	# print(weights_output)

	prev_delta_output = delta_output_layer
	prev_delta_hidden = delta_hidden_layer

### Testing ###

# Initialize variables
layer_hidden_init = np.dot(matrix_test_inputs, weights_hidden)
layer__hidden_output = logistic(layer_hidden_init)
output_layer_init = np.dot(layer_hidden_output, weights_output)
predict_output = logistic(output_layer_init)

for i in range(1059-train):
	print("Expected output: ")
	print(matrix_test_results[i])
	print("Predicted output: ")
	print(predict_output[i])
	print("\n")