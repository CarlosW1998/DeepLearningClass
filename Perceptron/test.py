from Perceptron import Perceptron
import numpy as np

ts_input = np.array([[0,0,1,0],
                         [1,1,1,0],
                         [1,0,1,1],
                         [0,1,1,1],
                         [0,1,0,1],
                         [1,1,1,1],
                         [0,0,0,0]])
ts_output = np.array([[0,1,1,0,0,1,0]]).T

model = Perceptron(4, 10)

model.train(ts_input, ts_output, 1000)

print(model.weigth)