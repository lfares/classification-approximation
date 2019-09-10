import csv
import numpy as np

# Declaration of functions
def sigmoid (x):
	return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
	return x * (1 - x)

# Define number of values to train (the rest of them will be to test)
train = 1000

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
            list_t.append(row[-(len(row)-2):])
        else:
            list_test_inputs.append(row[:-2])
            list_test_results.append(row[-(len(row)-2):])
        line_count += 1

print(list_inputs)
print(list_t)

# Change input and t value from lists to numpy matrix to train later
matrix_inputs = np.matrix(list_inputs)
matrix_t = np.matrix(list_t)

# Auxiliar variables
epochs = 100000
learningRate = 0.1

