import numpy as np

class Perceptron():
    def __init__(self, weigth, learnig_rate):
        self.weigth = np.random.rand(weigth, 1)
        self.learnig_rate = learnig_rate
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivate(self, x):
        return np.exp(x) / ((1+np.exp(-x))**2)
    

    def train(self, inputs, output, ephocs):
        delta_weights = np.zeros((len(self.weigth), len(inputs)))
        for i in range(ephocs):
            #FowardPass
            predict = np.dot(inputs, self.weigth)
            activation = self.sigmoid(predict)
            #Backwadr pass 
            for j in range(len(inputs)):
                #lost = (activation[j] - output[i])**2
                lost_prime = 2*(activation[j] - output[j])
                for z in range(len(self.weigth)):
                    delta_weights[z][j] = lost_prime * inputs[j][z] * self.sigmoid_derivate(predict[j])
            delta_avg = np.array([np.average(delta_weights, axis=1)]).T
            self.weigth = self.weigth - delta_avg * self.learnig_rate
    
    def predict(self, x):
        return self.sigmoid(np.dot(x, self.weigth))