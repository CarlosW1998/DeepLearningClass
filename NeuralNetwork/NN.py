import numpy as np

class NeuralNetwork:
    def __init__(self, input_shape, neurons, learning_rate):
        self.wights = []
        self.wights.append(np.random.rand(input_shape, neurons))
        self.wights.append(np.random.rand(neurons, 1))
        self.baias = np.zeros(neurons)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


