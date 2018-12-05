import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand((y, 1)) for y in self.sizes[1:]]
        self.weights = [np.random.rand(y, x) for x, y in zip(sizes[: -1], sizes[1:])]

    def feed_forward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)

    def back_propagation(self):
        pass

