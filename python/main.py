import numpy as np
import random


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, trainingData, epochs, miniBatchSize, eta, testData=None):
        if testData:
            nTest = len(testData)
        nData = len(trainingData)
        for epoch in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [
                trainingData[i, i+miniBatchSize]
                for i in range(0, nData, miniBatchSize)
            ]
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)
            if(testData):
                print("Epoch {0}: {1} / {2}"
                      .format(epoch, self.evaluate(testData), nTest))
            else:
                print("Epoch {} complete".format(epoch))

    def updateMiniBatch(self, miniBatch, eta):
        biasGradients = [np.zeros(b.shape) for b in self.biases]
        weightGradients = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            deltaBiasGradients, deltaWeightGradients = self.backprop(x, y)
            biasGradients = [previusTmpBiasGradient+deltaBiasGradient
                             for previusTmpBiasGradient, deltaBiasGradient
                             in zip(biasGradients, deltaBiasGradients)]
            weightGradients = [previusTmpWeightGradient+deltaWeightGradient
                               for previusTmpWeightGradient, deltaWeightGradient
                               in zip(weightGradients, deltaWeightGradients)]
        m = len(miniBatch)
        self.bias = [previusBias-(eta/m)*biasGradient
                     for previusBias, biasGradient
                     in zip(self.bias, biasGradients)]
        self.weights = [previusWeight-(eta/m)*weightGradient
                        for previusWeight, weightGradient
                        in zip(self.weights, weightGradients)]

    def backprop(x, y):
        return


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
