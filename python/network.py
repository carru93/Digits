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
                               for previusTmpWeightGradient,
                               deltaWeightGradient
                               in zip(weightGradients, deltaWeightGradients)]
        m = len(miniBatch)
        self.bias = [previusBias-(eta/m)*biasGradient
                     for previusBias, biasGradient
                     in zip(self.bias, biasGradients)]
        self.weights = [previusWeight-(eta/m)*weightGradient
                        for previusWeight, weightGradient
                        in zip(self.weights, weightGradients)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y)*sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
