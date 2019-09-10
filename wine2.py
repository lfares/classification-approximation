import numpy as np
import csv

# Declaration of functions
def logistic(x):
    return .5 * (1 + np.tanh(.5 * x))

def logistic_derivative(x):
    return logistic(x)*(1 - logistic(x))

#Pre-Processing
wines = []
length = 0
with open("winequality.csv", "r") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		length += 1
		wines.append(row)
wines = np.array(wines)
wines = np.delete(wines, 0, 1)
aux = wines[:,[11]]
wines = np.delete(wines, 11, 1)
wines = np.array(wines).astype(np.float)
wines = wines/wines.max(axis=0)

qualities = []
for label in aux:
	if label == 'Bad':
		qualities.append([1., 0., 0.])
	if label == 'Mid':
		qualities.append([0., 1., 0.])
	if label == 'Good':
		qualities.append([0., 0., 1.])
qualities = np.array(qualities)

#Auxiliar Variables
ratiosRange = [3, 4, 5, 6]
epochsRange = [5000, 7500, 10000]
learningRateRange = [0.05, 0.075, 0.1]
momentumRange = [0., 0.25, 0.5]
entries = len(wines[0])
hidden1_n = entries
hidden2_n = int(entries/2)

for ratio in ratiosRange:
	#Train and Test Sets
	testLen = int(length/ratio)
	test = np.random.random_integers(0, length-1, testLen)
	matrixTest = wines[test, :]
	matrixTestT = qualities[test, :]
	matrixTrain= np.delete(wines, test, 0)
	matrixTrainT = np.delete(qualities, test, 0)
	for momentum in momentumRange:
		for learningRate in learningRateRange:
			for epochs in epochsRange:
				# Initialization of random vectors for weights
				weightsHidden1 = np.random.uniform(low=-0.1, high=0.1, size=(entries, hidden1_n))
				weightsHidden2 = np.random.uniform(low=-0.1, high=0.1, size=(hidden1_n, hidden2_n))
				weightsOutput = np.random.uniform(low=-0.1, high=0.1, size=(hidden2_n, 3))
				prevDeltaHidden1 = np.zeros(shape=(entries, hidden1_n))
				prevDeltaHidden2 = np.zeros(shape=(hidden1_n, hidden2_n))
				prevDeltaOut = np.zeros(shape=(hidden2_n, 3))
				#Training
				for _ in range(epochs):
					#Forward Propagation
					layerHidden1Init = np.dot(matrixTrain,weightsHidden1)
					layerHidden1Output = logistic(layerHidden1Init)

					layerHidden2Init = np.dot(layerHidden1Output,weightsHidden2)
					layerHidden2Output = logistic(layerHidden2Init)

					outputLayerInit = np.dot(layerHidden2Output,weightsOutput)
					predictOutput = logistic(outputLayerInit)

					#Backpropagation
					errorOutputLayer = matrixTrainT - predictOutput
					deltaOutputLayer = np.dot(layerHidden2Output.T, errorOutputLayer*logistic_derivative(predictOutput))*learningRate
					
					errorHiddenLayer2 = errorOutputLayer.dot(weightsOutput.T)
					deltaHiddenLayer2 = np.dot(layerHidden1Output.T, errorHiddenLayer2*logistic_derivative(layerHidden2Output))*learningRate

					errorHiddenLayer1 = errorHiddenLayer2.dot(weightsHidden2.T)
					deltaHiddenLayer1 = np.dot(matrixTrain.T, errorHiddenLayer1*logistic_derivative(layerHidden1Output))*learningRate

					#Updating weights
					weightsOutput += deltaOutputLayer + momentum*prevDeltaOut
					weightsHidden2 += deltaHiddenLayer2 + momentum*prevDeltaHidden2
					weightsHidden1 += deltaHiddenLayer1 + momentum*prevDeltaHidden1

					prevDeltaOut = deltaOutputLayer
					prevDeltaHidden2 = deltaHiddenLayer2
					prevDeltaHidden1 = deltaHiddenLayer1

				#Test
				layerHidden1Init = np.dot(matrixTest,weightsHidden1)
				layerHidden1Output = logistic(layerHidden1Init)
				layerHidden2Init = np.dot(layerHidden1Output,weightsHidden2)
				layerHidden2Output = logistic(layerHidden2Init)
				outputLayerInit = np.dot(layerHidden2Output,weightsOutput)
				predictOutput = np.rint(logistic(outputLayerInit))

				accuracy = 0
				for i in range(testLen):
					#print(str(predictOutput[i]) + " " + str(matrixTestT[i]))
					if np.array_equal(predictOutput[i], matrixTestT[i]):
						accuracy += 1

				accuracy = (accuracy/testLen)*100
				print("Ratio: "+str(ratio-1)+":1 Learning Rate: "+str(learningRate)+" Momentum: "+str(momentum)+" Cycles: "+str(epochs)+" Accuracy: "+str(np.around(accuracy, 3))+"%")