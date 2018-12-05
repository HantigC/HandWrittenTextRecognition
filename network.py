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

    def back_propagation(self, x, y):
        raise NotImplementedError("Implement back propagation method")

    def evaluate(self, test_data):
        test_result = [(self.feed_forward(x), y) for (x, y) in test_data]
        _test_result = sum(int(x == y) for (x, y) in test_result)
        return float(_test_result) / float(len(test_result))

    def learn(self, batch, learning_rate):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            grad_b, grad_w = self.back_propagation(x, y)
            delta_b = [b + db for b, db in zip(grad_b, delta_b)]
            delta_w = [w + dw for w, dw in zip(grad_w, delta_w)]
        self.weights = [w - (learning_rate / len(batch)) * dw for w, dw in zip(self.weights, delta_w)]
        self.biases = [b - (learning_rate) / len(batch) * db for b, db in zip(self.biases, delta_b)]